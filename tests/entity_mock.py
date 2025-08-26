class EntityMock:
    def __init__(self, name, parts=None):
        self.name = name
        self.isDefinedBy = [name + "-def"]
        self.INDIRECT_hasPart = parts
        self.iri = f"http://edugraph.io/edu#{name}"
