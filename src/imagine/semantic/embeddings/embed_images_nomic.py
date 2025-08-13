import os
import glob
import argparse
import json
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModel
import math


def generate_nomic_embeddings(directory_path: str, batch_size: int = 8):
    """
    Generates embeddings for all .png files in the specified directory
    using the nomic-embed-multimodal-3b model and stores them as line-separated JSON arrays.
    """
    try:
        # 1. Automatic device selection (GPU for performance)
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        print(f"Using device: {device}")

        # It's good practice to specify the cache directory for models
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")

        processor = AutoProcessor.from_pretrained("nomic-ai/nomic-embed-multimodal-3b", trust_remote_code=True,
                                                  cache_dir=cache_dir)
        model = AutoModel.from_pretrained("nomic-ai/nomic-embed-multimodal-3b", trust_remote_code=True,
                                          cache_dir=cache_dir)

        model.to(device)
        model.eval()

        png_files = glob.glob(os.path.join(directory_path, "*.png"))

        if not png_files:
            print(f"No .png files found in the directory: {directory_path}")
            return

        # 2. Simplified and more robust output path
        output_dir = os.path.join(directory_path, "out")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory will be: {output_dir}")

        print(f"Found {len(png_files)} .png files. Generating embeddings in batches of {batch_size}...")

        total_batches = math.ceil(len(png_files) / batch_size)

        # 3. Process images in batches for efficiency
        for i in range(0, len(png_files), batch_size):
            batch_files = png_files[i:i + batch_size]
            batch_images = []

            current_batch_num = (i // batch_size) + 1
            print(f"Processing batch {current_batch_num}/{total_batches}...")

            for file_path in batch_files:
                try:
                    image = Image.open(file_path).convert("RGB")
                    batch_images.append(image)
                except Exception as e:
                    print(f"Could not open or process {os.path.basename(file_path)}: {e}")

            if not batch_images:
                continue

            try:
                inputs = processor(images=batch_images, return_tensors="pt").to(device)

                with torch.no_grad():
                    outputs = model(**inputs)

                # The primary output for images is `image_embeds`
                if hasattr(outputs, 'image_embeds') and outputs.image_embeds is not None:
                    embeddings = outputs.image_embeds.cpu().tolist()
                else:
                    # Fallback if the model output structure is different
                    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().tolist()

                # Save each embedding from the batch to its own file
                for file_path, embedding in zip(batch_files, embeddings):
                    output_filename = os.path.splitext(os.path.basename(file_path))[0] + ".jsonl"
                    output_filepath = os.path.join(output_dir, output_filename)
                    with open(output_filepath, 'w') as f:
                        json.dump(embedding, f)
                        f.write('\n')

                print(f"  Saved {len(batch_files)} embeddings from batch {current_batch_num}.")

            except Exception as e:
                print(f"Error processing batch starting with {os.path.basename(batch_files[0])}: {e}")

    except Exception as e:
        print(f"An error occurred during model loading or initialization: {e}")
        print("Please try updating your libraries with: pip install --upgrade transformers accelerate")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for .png files using nomic-embed-multimodal-3b.")
    parser.add_argument("directory", type=str, help="Path to the directory containing .png files.")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of images to process at once.")
    args = parser.parse_args()

    generate_nomic_embeddings(args.directory, args.batch_size)