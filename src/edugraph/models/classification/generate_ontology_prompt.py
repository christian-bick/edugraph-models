from edugraph.ontology_loader import load_ontology_owlready2
from edugraph.ontology_serializer_text import build_taxonomy
from edugraph.ontology_util import OntologyUtil


def generate_from_template_file(template_file, onto):
    print(f"Reading template from file: {template_file}")
    with open(template_file, "r") as f:
        template = f.read()
    return generate_from_template(template, onto)


def generate_from_template(template, onto):
    o = OntologyUtil(onto)
    area_taxonomy = build_taxonomy("Area", onto.Area.iri, o.list_root_entities(onto.Area))
    scope_taxonomy = build_taxonomy("Scope", onto.Scope.iri, o.list_root_entities(onto.Scope))
    ability_taxonomy = build_taxonomy("Ability", onto.Ability.iri, o.list_root_entities(onto.Ability))
    return template.format(area_taxonomy, scope_taxonomy, ability_taxonomy)


if __name__ == "__main__":
    ontology = load_ontology_owlready2(
        "https://github.com/christian-bick/edugraph-ontology/releases/download/v0.4.0/core-ontology.rdf")
    input_file = "./prompts/classify_with_taxonomy_v1.txt"
    output_file = "./out/entity_classification_instruction.txt"
    prompt = generate_from_template_file(input_file, ontology)
    print(f"Writing prompt to file: {output_file}")
    with open(output_file, "w", encoding="UTF-8") as f:
        prompt = f.write(prompt)
