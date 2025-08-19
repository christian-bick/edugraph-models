from imagine.semantic.ontology_util import *


def build_taxonomy(dimension_name, dimension_iri, entities):
    taxonomy = f"# {dimension_name} ({dimension_iri})\n\n"
    taxonomy += __build_outline_header()
    taxonomy += build_outline([1], entities)
    taxonomy += __build_definition_header()
    taxonomy += build_definitions([1], entities)
    return taxonomy


def build_outline(hierarchy, entities):
    depth = len(hierarchy) - 1
    content = ""

    for index, entity in enumerate(entities):
        hierarchy[depth] = index + 1
        content += __build_outline_item(hierarchy, entity)

        if not is_leaf_entity(entity):
            entity_parts = parts_of_entity(entity)
            new_hierarchy = hierarchy + [1]
            content += build_outline(new_hierarchy, entity_parts)

    return content


def build_definitions(hierarchy, entities):
    depth = len(hierarchy) - 1
    content = ""

    for index, entity in enumerate(entities):
        hierarchy[depth] = index + 1
        content += __build_definition_item(hierarchy, entity)

        if not is_leaf_entity(entity):
            entity_parts = parts_of_entity(entity)
            new_hierarchy = hierarchy + [1]
            content += build_definitions(new_hierarchy, entity_parts)

    return content


def __build_outline_item(hierarchy, entity):
    return f"{__build_outline_index(hierarchy)} {natural_name_of_entity(entity)} ({entity.iri})\n"


def __build_definition_item(hierarchy, entity):
    entity_definition = definition_of_entity(entity)
    definition = ""

    if len(entity_definition) > 0:
        definition += f"### {__build_outline_item(hierarchy, entity)}"
        definition += "\n{0}\n\n".format(entity_definition)

    return definition


def __build_outline_header():
    return "## Outline\n\n"


def __build_definition_header():
    return "\n\n## Definitions\n\n"


def __build_outline_index(hierarchy):
    level_string = reduce(lambda tail, head: tail + str(head) + '.', hierarchy, "")
    return level_string[:-1]
