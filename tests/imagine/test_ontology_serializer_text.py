import pytest

from imagine.ontology_serializer_text import *
from tests.entity_mock import EntityMock

# Updated expected strings to include the mock IRI

expected_outline = (
"""1 e1 (http://edugraph.io/edu#e1)
1.1 e1-e1 (http://edugraph.io/edu#e1-e1)
1.1.1 e1-e1-e1 (http://edugraph.io/edu#e1-e1-e1)
1.1.2 e1-e1-e2 (http://edugraph.io/edu#e1-e1-e2)
1.2 e1-e2 (http://edugraph.io/edu#e1-e2)
"""
)

expected_definitions = (
"""### 1 e1 (http://edugraph.io/edu#e1)

e1-def

### 1.1 e1-e1 (http://edugraph.io/edu#e1-e1)

e1-e1-def

### 1.1.1 e1-e1-e1 (http://edugraph.io/edu#e1-e1-e1)

e1-e1-e1-def

### 1.1.2 e1-e1-e2 (http://edugraph.io/edu#e1-e1-e2)

e1-e1-e2-def

### 1.2 e1-e2 (http://edugraph.io/edu#e1-e2)

e1-e2-def

"""
)

# Updated expected taxonomy to match the new function output format
expected_taxonomy = (
"""# Areas (http://edugraph.io/edu#Area)

## Outline

{0}

## Definitions

{1}"""
).format(expected_outline, expected_definitions)


class TestOntologySerializerText:

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

    def test_build_outline(self, entities):
        outline = build_outline([1], entities)
        assert outline == expected_outline

    def test_build_definitions(self, entities):
        definitions = build_definitions([1], entities)
        assert definitions == expected_definitions

    def test_build_taxonomy(self, entities):
        # Added the missing IRI argument
        taxonomy = build_taxonomy('Areas', "http://edugraph.io/edu#Area", entities)
        assert taxonomy == expected_taxonomy