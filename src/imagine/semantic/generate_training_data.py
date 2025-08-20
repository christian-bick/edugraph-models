
import argparse
import json
import os

from merge_meta_files import merge_meta_from_gcs

def generate_training_file(bucket_name, output_file):
    """
    Generates a training data file by fetching and merging data from GCS.
    """
    print(f"Generating training data from bucket: {bucket_name}")
    
    # 1. Get intermediate data from the merge script
    intermediate_data = merge_meta_from_gcs(bucket_name)

    if not intermediate_data:
        print("No intermediate data found. Output file will not be created.")
        return

    # 2. Perform the final mapping to the training format
    print(f"Mapping {len(intermediate_data)} items to the final training format...")
    final_training_data = []
    for item in intermediate_data:
        # The 'input' is now directly the 'questionDoc' field (which contains the full GCS URI)
        final_item = {
            'input': item['questionDoc'],
            'output': item['labels']
        }
        final_training_data.append(final_item)

    # 3. Write the final data to the output file
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_training_data, f, indent=2)
        print(f"Successfully wrote {len(final_training_data)} items to {output_file}")
    except IOError as e:
        print(f"Error writing to output file {output_file}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a training dataset from meta.json files in a GCS bucket."
    )
    parser.add_argument(
        "bucket_name",
        type=str,
        help="The name of the GCS bucket to process."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./temp/training_data.json",
        help="The path for the output training data file."
    )
    
    args = parser.parse_args()
    
    try:
        generate_training_file(args.bucket_name, args.output_file)
    except (RuntimeError, FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        print(f"Error: {e}")
        exit(1)
