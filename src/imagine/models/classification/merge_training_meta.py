
import argparse
import json
import os
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()

def merge_meta_from_gcs(bucket_name):
    """
    Parses folders in a GCS bucket, reads meta.json, and creates a list of
    intermediate data items.

    This function is strict: any missing file or malformed data will
    raise an exception.
    """
    print(f"Starting to process bucket: {bucket_name}")
    
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
    except Exception as e:
        raise RuntimeError(f"Error initializing GCS client: {e}") from e

    merged_data = []
    processed_folders = set()

    blobs = list(bucket.list_blobs())
    if not blobs:
        print(f"Bucket '{bucket_name}' is empty or does not exist.")
        return []
        
    print(f"Found {len(blobs)} total objects. Identifying folders...")

    for blob in blobs:
        folder_name = os.path.dirname(blob.name)
        
        if folder_name and folder_name not in processed_folders:
            processed_folders.add(folder_name)
            meta_file_path = f"{folder_name}/meta.json"
            
            meta_blob = bucket.blob(meta_file_path)
            
            if meta_blob.exists():
                print(f"  Found and processing: gs://{bucket_name}/{meta_file_path}")
                
                content = meta_blob.download_as_text()
                meta_items = json.loads(content)
                
                for item in meta_items:
                    if 'questionDoc' not in item or 'labels' not in item:
                        raise ValueError(f"Item in {meta_file_path} is missing 'questionDoc' or 'labels' fields.")

                    question_doc_filename = item['questionDoc']
                    question_doc_blob_name = f"{folder_name}/{question_doc_filename}"
                    
                    question_blob = bucket.blob(question_doc_blob_name)
                    if not question_blob.exists():
                        raise FileNotFoundError(f"Referenced questionDoc file not found: gs://{bucket_name}/{question_doc_blob_name}")

                    gcs_uri = f"gs://{bucket_name}/{question_doc_blob_name}"

                    # Intermediate representation with full GCS URI
                    mapped_item = {
                        'questionDoc': gcs_uri,
                        'labels': item['labels']
                    }
                    merged_data.append(mapped_item)

    if not merged_data:
        print("No data was merged.")

    return merged_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge meta.json files from a GCS bucket."
    )
    parser.add_argument(
        "bucket_name",
        type=str,
        help="The name of the GCS bucket to process."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./temp/meta-prepared.json",
        help="The path to the output file for the merged data."
    )
    
    args = parser.parse_args()
    
    try:
        data = merge_meta_from_gcs(args.bucket_name)

        if data:
            output_file = args.output_file
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            print(f"Successfully wrote {len(data)} items to {output_file}")

    except (RuntimeError, FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        print(f"Error: {e}")
        exit(1)
