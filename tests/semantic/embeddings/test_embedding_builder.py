import pytest
from dotenv import load_dotenv

from imagine.ontology_loader import load_from_path
from imagine.ontology_util import OntologyUtil
from imagine.semantic.embeddings.embedding_builder import generate_embedding_input
from tests.entity_mock import EntityMock

onto = load_from_path("./tests/test_data/test-ontology.rdf")
onto_util = OntologyUtil(onto)

load_dotenv()

class TestEmbeddingBuilder:

    @pytest.fixture
    def entities(self):
        return [EntityMock('e1', [
            EntityMock('e1-e1', [
                EntityMock('e1-e1-e1'),
                EntityMock('e1-e1-e2')
            ]), EntityMock(
                'e1-e2'
            )]
        )]

    def test_generate_embedding_input(self, entities):
        input_map = generate_embedding_input(entities)
        assert input_map == {
            'e1': 'e1-def',
            'e1-e1': 'e1-e1-def',
            'e1-e2': 'e1-e2-def',
            'e1-e1-e1': 'e1-e1-e1-def',
            'e1-e1-e2': 'e1-e1-e2-def'
        }