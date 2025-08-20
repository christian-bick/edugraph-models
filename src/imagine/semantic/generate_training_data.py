import argparse
import json
import os

from merge_meta_files import merge_meta_from_gcs


def _get_cached_or_fresh_data(bucket_name, no_cache):
    """
    Handles the logic of reading from a local cache or fetching fresh data
    from GCS and updating the cache.
    """
    cache_file = "./temp/meta_data_cache.json"

    # 1. Check for and load from cache if appropriate
    if not no_cache and os.path.exists(cache_file):
        print(f"Cache hit. Loading intermediate data from {cache_file}")
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Warning: Could not read cache file {cache_file}. Error: {e}. Fetching fresh data.")
            # Fall through to fetch fresh data

    # 2. If no cache was used or cache failed, fetch from GCS
    if no_cache:
        print("Cache ignored (--no-cache). Fetching fresh data from GCS...")
    else:
        print(f"Cache miss. Fetching fresh data from GCS...")

    intermediate_data = merge_meta_from_gcs(bucket_name)

    # 3. Update the cache
    if intermediate_data:
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(intermediate_data, f, indent=2)
            print(f"Cache updated with {len(intermediate_data)} items at {cache_file}")
        except IOError as e:
            print(f"Warning: Could not write to cache file {cache_file}. Error: {e}")

    return intermediate_data


def generate_training_file(bucket_name, output_file, no_cache=False):
    """
    Generates a training data file by fetching and merging data from GCS,
    using a local cache to speed up subsequent runs.
    """
    # 1. Get data using the caching helper function
    intermediate_data = _get_cached_or_fresh_data(bucket_name, no_cache)

    if not intermediate_data:
        print("No intermediate data found. Output file will not be created.")
        return

    # 2. Perform the final mapping to the training format
    print(f"Mapping {len(intermediate_data)} items to the final training format...")
    final_training_data = []
    for item in intermediate_data:
        # Ensure 'labels' is a dictionary
        labels = item.get('labels', {})
        if not isinstance(labels, dict):
            print(f"Warning: 'labels' for item is not a dictionary. Skipping item. Labels: {labels}")
            continue

        # Map the labels to the desired output structure
        output_labels = {}
        for key, iri_list in labels.items():
            if isinstance(iri_list, list):
                output_labels[key] = [{'iri': iri} for iri in iri_list]
            else:
                print(f"Warning: Value for key '{key}' in labels is not a list. Skipping key. Value: {iri_list}")

        user_content = [
            {
                "text": "Classify the educational concepts in this document."},
            {
                "fileData": {
                    "mimeType": "application/pdf",
                    "fileUri": item['questionDoc']
                }
            }
        ]

        final_item = {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "model", "content": [{"text": json.dumps(output_labels)}]}
            ]
        }
        final_training_data.append(final_item)

    # 3. Write the final data to the output file
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in final_training_data:
                f.write(json.dumps(item) + '\n')
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
        default="./temp/training_data.jsonl",
        help="The path for the output training data file."
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore any existing cache and fetch fresh data from GCS."
    )

    args = parser.parse_args()

    try:
        generate_training_file(args.bucket_name, args.output_file, args.no_cache)
    except (RuntimeError, FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        print(f"Error: {e}")
        exit(1)
