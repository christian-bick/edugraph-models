import ctypes
from io import BytesIO
from uuid import uuid4

from flask import request, jsonify
from google import genai
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import Namespace
from google.genai.types import UploadFileConfig

from imagine.api import app
from imagine.config import GCP_PROJECT, GCP_LOCATION, VECTOR_SEARCH_INDEX_ENDPOINT, VECTOR_SEARCH_INDEX, \
    PUBLIC_DOCS_IMAGE_URL, PUBLIC_DOCS_ORIGINAL_URL
from imagine.semantic.classification_cache import ClassificationCache
from imagine.semantic.classifiers.merged_classifier import MergedClassifier
from imagine.semantic.classifiers.strategies.classifier_split_gemini_with_serialized_taxonomies_v1 import \
    ClassifierSplitGeminiWithSerializedTaxonomiesV1
from imagine.semantic.embeddings.embedder_google import GoogleMultiModalEmbedder
from imagine.semantic.embeddings.find_by_image import query_vector_search_index
from imagine.semantic.ontology_loader import load_from_path
from imagine.semantic.ontology_serializer import serialize_entity_tree, serialize_entities_with_names, \
    serialize_entity_tree_with_parent_relations
from imagine.semantic.ontology_util import OntologyUtil

onto_ttl = "./core-ontology.ttl"
onto_path = "./core-ontology.rdf"
onto = load_from_path(onto_path)
onto_util = OntologyUtil(onto)

classification_cache = ClassificationCache()

root_areas = onto_util.list_root_entities(onto.Area)
root_abilities = onto_util.list_root_entities(onto.Ability)
root_scopes = onto_util.list_root_entities(onto.Scope)


@app.route("/")
def root():
    return "OK"


@app.route("/classify", methods=["POST"])
def classify():
    file = request.files['file']
    file_name = request.values['name']
    file_mimetype = file.mimetype
    file_content = BytesIO(file.stream.read())

    return classify_upload(file_name, file_mimetype, file_content)


@app.route("/search", methods=["POST"])
def search():
    file = request.files['file']
    file_name = request.values['name']
    file_content = file.stream.read()

    print(file_name)

    neighbors = find_similar(file_name, file_content)
    result = list(map(lambda x: { "content": { "preview": f"{PUBLIC_DOCS_IMAGE_URL}/{x.id}", "original": f"{PUBLIC_DOCS_ORIGINAL_URL}/{x.id}" }}, neighbors))

    return result


@app.route("/ontology", methods=["GET"])
def ontology():
    return jsonify({
        "taxonomy": {
            "areas": serialize_entity_tree(root_areas, "hasPartArea"),
            "abilities": serialize_entity_tree(root_abilities, "hasPartAbility"),
            "scopes": serialize_entity_tree(root_scopes, "hasPartScope")
        }})


def find_similar(file_name, query_blob):
    project = GCP_PROJECT
    location = GCP_LOCATION

    embedder = GoogleMultiModalEmbedder(
        model_name="multimodalembedding@001"
    )

    query_embeddings = embedder.embed_document(file_name, query_blob)

    neighbors = query_vector_search_index(
        project=project,
        location=location,
        index_endpoint=VECTOR_SEARCH_INDEX_ENDPOINT,
        deployed_index=VECTOR_SEARCH_INDEX,
        query_embedding=query_embeddings[0]["embedding"],
        filter=[Namespace(
            name="type",
            allow_tokens=["material"]
        )]
    )

    return neighbors


def classify_upload(file_name, file_mimetype, file_content: BytesIO):
    if file_name is None or file_name == '':
        name = str(uuid4())
    else:
        name = file_name

    name = str(ctypes.c_size_t(hash(name)).value)

    result = classification_cache.get(name)

    if result is not None:
        app.logger.info('classification used from cache')
    else:
        app.logger.info('classification starting')
        mime_type = file_mimetype

        file = None
        client = genai.Client()
        try:
            file = client.files.get(name=name)
            app.logger.info('file %s retrieved from gemini', name)
        except:
            app.logger.info('file %s not in gemini', name)

        if file is None:
            file = client.files.upload(
                file=file_content,
                config=UploadFileConfig(
                    name=name,
                    mime_type=mime_type)
            )
            app.logger.info('file %s added to gemini', name)

        classifier = MergedClassifier(ClassifierSplitGeminiWithSerializedTaxonomiesV1(onto))
        classification = classifier.classify_content(file)
        classified_area = getattr(onto, classification["Area"][0])

        result = jsonify({
            "classification": {
                "areas": serialize_entities_with_names(classification["Area"]),
                "abilities": serialize_entities_with_names(classification["Ability"]),
                "scopes": serialize_entities_with_names(classification["Scope"]),
            },
            "expansion": {
                "areas": serialize_entity_tree_with_parent_relations([classified_area], "expandsArea", "partOfArea"),
            }
        })

        classification_cache.update(name, result)

    return result
