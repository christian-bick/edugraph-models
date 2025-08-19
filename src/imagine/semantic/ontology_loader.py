import os
import urllib.request
from urllib.parse import urlparse

from owlready2 import get_ontology

BASE_IRI = "http://edugraph.io/edu#"
TEMP_DIR = "temp"


def load_from_path(path):
    if path.startswith("https"):
        url_parts = urlparse(path).path.split('/')
        if len(url_parts) > 2:
            version = url_parts[-2]
            filename = url_parts[-1]

            if not os.path.exists(TEMP_DIR):
                os.makedirs(TEMP_DIR)

            basename, ext = os.path.splitext(filename)
            cached_filename = f"{basename}-{version}{ext}"
            cached_filepath = os.path.join(TEMP_DIR, cached_filename)

            if os.path.exists(cached_filepath):
                print(f"Using cached ontology from: {cached_filepath}")
                path = cached_filepath
            else:
                print(f"Downloading ontology from {path}...")
                try:
                    urllib.request.urlretrieve(path, cached_filepath)
                    print(f"Cached ontology at: {cached_filepath}")
                    path = cached_filepath
                except Exception as e:
                    print(f"Error downloading ontology: {e}")
                    # Fallback to original path URL

    # For local files, owlready2 needs a file URI
    if not path.startswith("http"):
        path = f"file://{os.path.abspath(path)}"

    onto = get_ontology(path).load()
    onto.base_iri = BASE_IRI
    return onto