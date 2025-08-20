
import argparse
import json
import os
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()

def merge_meta_from_gcs(bucket_name, output_file):
    """
    Parses folders in a GCS bucket, reads meta.json from each,
    and merges them into a single file, keeping only specified fields.
    """
    print(f"Starting to process bucket: {bucket_name}")
    
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
    except Exception as e:
        print(f"Error initializing GCS client: {e}")
        return

    merged_data = []
    processed_folders = set()

    # List all blobs to identify folders
    blobs = list(bucket.list_blobs())
    if not blobs:
        print(f"Bucket '{bucket_name}' is empty or does not exist.")
        return
        
    print(f"Found {len(blobs)} total objects. Identifying folders...")

    for blob in blobs:
        # Extract the folder name from the blob path
        folder_name = os.path.dirname(blob.name)
        
        if folder_name and folder_name not in processed_folders:
            processed_folders.add(folder_name)
            meta_file_path = f"{folder_name}/meta.json"
            
            meta_blob = bucket.blob(meta_file_path)
            
            if meta_blob.exists():
                print(f"  Found and processing: gs://{bucket_name}/{meta_file_path}")
                try:
                    content = meta_blob.download_as_text()
                    meta_items = json.loads(content)
                    
                    for item in meta_items:
                        if 'questionDoc' in item and 'labels' in item:
                            mapped_item = {
                                'questionDoc': item['questionDoc'],
                                'labels': item['labels']
                            }
                            merged_data.append(mapped_item)
                        else:
                            print(f"    Warning: Skipping item in {meta_file_path} due to missing 'questionDoc' or 'labels'.")
                
                except json.JSONDecodeError:
                    print(f"    Error: Could not decode JSON from {meta_file_path}.")
                except Exception as e:
                    print(f"    Error processing file {meta_file_path}: {e}")
            else:
                # This is expected if a folder doesn't have a meta.json
                pass


    if not merged_data:
        print("No data was merged. The output file will not be created.")
        return

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    # Write the merged data to the output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2)
        print(f"Successfully merged {len(merged_data)} items into {output_file}")
    except IOError as e:
        print(f"Error writing to output file {output_file}: {e}")


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
    
    # Assuming the script is run from the project root
    merge_meta_from_gcs(args.bucket_name, args.output_file)
