import os
import glob
import argparse
import json
import torch
import math
from PIL import Image

# 1. Import the correct custom classes from colpali and transformers
try:
    from colpali_engine.models import BiQwen2_5, BiQwen2_5_Processor
    from transformers.utils.import_utils import is_flash_attn_2_available
except ImportError:
    print("Error: 'colpali' library not found.")
    print("Please install it by running: pip install git+https://github.com/illuin-tech/colpali.git")
    exit()


def generate_nomic_embeddings(directory_path: str, batch_size: int = 8):
    """
    Generates embeddings for all .png files in the specified directory
    using nomic-embed-multimodal-3b with the correct colpali engine.
    """
    try:
        # --- Model Loading ---
        model_name = "nomic-ai/nomic-embed-multimodal-3b"

        # Check for CUDA device
        if not torch.cuda.is_available():
            print("Error: CUDA is not available. This model requires a GPU.")
            return
        device = "cuda:0"
        print(f"Using device: {device}")

        # Check for Flash Attention 2 for optimized performance
        attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else None
        if attn_implementation:
            print("Flash Attention 2 is available. Using for model loading.")
        else:
            print("Flash Attention 2 not found. Using default attention mechanism.")

        print("Loading model... (This may take a moment)")
        # 2. Load the model using the specific BiQwen2_5 class and recommended settings
        model = BiQwen2_5.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,  # bfloat16 is efficient for modern GPUs
            device_map=device,
            attn_implementation=attn_implementation,
        ).eval()

        # 3. Load the processor using the specific BiQwen2_5_Processor class
        processor = BiQwen2_5_Processor.from_pretrained(model_name)
        print("Model and processor loaded successfully.")

        # --- Image Processing ---
        png_files = glob.glob(os.path.join(directory_path, "*.png"))
        if not png_files:
            print(f"No .png files found in the directory: {directory_path}")
            return

        output_dir = os.path.join(directory_path, "out")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory will be: {output_dir}")

        print(f"Found {len(png_files)} .png files. Generating embeddings in batches of {batch_size}...")
        total_batches = math.ceil(len(png_files) / batch_size)

        for i in range(0, len(png_files), batch_size):
            batch_files = png_files[i:i + batch_size]
            batch_images = []
            current_batch_num = (i // batch_size) + 1
            print(f"Processing batch {current_batch_num}/{total_batches}...")

            for file_path in batch_files:
                try:
                    # The processor expects PIL images
                    image = Image.open(file_path).convert("RGB")
                    batch_images.append(image)
                except Exception as e:
                    print(f"Could not open or process {os.path.basename(file_path)}: {e}")

            if not batch_images:
                continue

            try:
                # 4. Process images using the new processor's method
                inputs = processor.process_images(batch_images).to(model.device)

                with torch.no_grad():
                    # The model's forward pass now directly gives embeddings
                    embeddings = model(**inputs)

                # Save each embedding from the batch to its own file
                for file_path, embedding in zip(batch_files, embeddings):
                    output_filename = os.path.splitext(os.path.basename(file_path))[0] + ".jsonl"
                    output_filepath = os.path.join(output_dir, output_filename)
                    with open(output_filepath, 'w') as f:
                        # Convert tensor to list for JSON serialization
                        json.dump(embedding.cpu().tolist(), f)
                        f.write('\n')

                print(f"  Saved {len(batch_files)} embeddings from batch {current_batch_num}.")

            except Exception as e:
                print(f"Error processing batch starting with {os.path.basename(batch_files[0])}: {e}")

    except Exception as e:
        print(f"An error occurred during model loading or initialization: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for .png files using nomic-embed-multimodal-3b.")
    parser.add_argument("directory", type=str, help="Path to the directory containing .png files.")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of images to process at once.")
    args = parser.parse_args()
    generate_nomic_embeddings(args.directory, args.batch_size)