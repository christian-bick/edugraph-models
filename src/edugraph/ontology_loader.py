import os
import urllib.request
from urllib.parse import urlparse

from owlready2 import get_ontology
from rdflib import Graph

BASE_IRI = "http://edugraph.io/edu#"
TEMP_DIR = "temp"


def cache_url(url: str) -> str:
    """
    Caches a file from a URL to the local temp directory.
    It creates a versioned filename to avoid conflicts.
    If the file is already cached, it returns the local path.
    Otherwise, it downloads, caches, and then returns the path.

    Args:
        url (str): The URL of the file to cache.

    Returns:
        str: The local file path to the cached file.
    """
    if not url.startswith("http"):
        return url  # Assumed to be a local path already

    try:
        url_parts = urlparse(url).path.split('/')
        if len(url_parts) < 2:
            raise ValueError("URL path does not have enough parts to determine version.")

        version = url_parts[-2]
        filename = url_parts[-1]

        if not os.path.exists(TEMP_DIR):
            os.makedirs(TEMP_DIR)

        basename, ext = os.path.splitext(filename)
        cached_filename = f"{basename}-{version}{ext}"
        cached_filepath = os.path.join(TEMP_DIR, cached_filename)

        if os.path.exists(cached_filepath):
            print(f"Using cached file from: {cached_filepath}")
            return cached_filepath
        else:
            print(f"Downloading from {url}...")
            with urllib.request.urlopen(url, timeout=30) as response, open(cached_filepath, 'wb') as out_file:
                out_file.write(response.read())
            print(f"Cached file at: {cached_filepath}")
            return cached_filepath

    except Exception as e:
        print(f"Error during caching/download of {url}: {e}")
        print("Falling back to using the original URL directly.")
        return url


def load_ontology_owlready2(path: str):
    """
    Loads an ontology from a given path (URL or local) using owlready2.
    Uses caching for URL paths.
    """
    cached_path = cache_url(path)

    # For local files, owlready2 needs a file URI
    if not cached_path.startswith("http"):
        path_for_owl = f"file://{os.path.abspath(cached_path)}"
    else:
        path_for_owl = cached_path

    onto = get_ontology(path_for_owl).load()
    onto.base_iri = BASE_IRI
    return onto


def load_ontology_rdflib(url: str) -> Graph:
    """
    Loads an RDF graph from a given URL using rdflib.
    Uses caching for URL paths.
    """
    cached_path = cache_url(url)

    g = Graph()
    print(f"Attempting to load RDF data into rdflib Graph from: {cached_path}")
    try:
        # rdflib can often guess the format from the file extension
        g.parse(cached_path)
        print(f"Successfully loaded KG with {len(g)} triples.")
    except Exception as e:
        print(f"Failed to load or parse the graph with rdflib. Error: {e}")
        return Graph()  # Return an empty graph on failure
    return g
