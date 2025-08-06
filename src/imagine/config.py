GCP_PROJECT = "edugraph-438718"
GCP_LOCATION = "europe-west3"

GCP_BUCKET = "edugraph-embed"
GCP_BUCKET_PATH = "examples"

STORAGE_URL = f"https://storage.googleapis.com/{GCP_BUCKET}"

PUBLIC_DOCS_IMAGE_URL = f"{STORAGE_URL}/{GCP_BUCKET_PATH}/normalized"
PUBLIC_DOCS_ORIGINAL_URL = f"{STORAGE_URL}/{GCP_BUCKET_PATH}/normalized"

VECTOR_SEARCH_INDEX_ENDPOINT = "projects/575953891979/locations/europe-west3/indexEndpoints/6601907617818214400"
VECTOR_SEARCH_INDEX = "example_deploy_1749682778003"