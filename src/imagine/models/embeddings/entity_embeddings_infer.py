import onnxruntime
import torch
import numpy as np

import os

def embed_entities(entity_uris, model_path, data_path):
    """
    Loads an ONNX model and associated data to compute a pooled embedding for a list of entity URIs.

    Args:
        entity_uris (list[str]): A list of entity URIs to be pooled.
        model_path (str): Path to the ONNX model file.
        data_path (str): Path to the inference data file (.pt).

    Returns:
        numpy.ndarray: The pooled embedding vector.
    """
    if not os.path.exists(model_path) or not os.path.exists(data_path):
        print("Error: Model or data file not found.")
        print("Please run the training script to generate the model and data.")
        return None

    # 1. Load inference data
    inference_data = torch.load(data_path)
    x = inference_data['x']
    edge_index = inference_data['edge_index']
    edge_type = inference_data['edge_type']
    entity_map = inference_data['entity_map']

    # 2. Map URIs to indices
    pool_indices_list = [entity_map[uri] for uri in entity_uris if uri in entity_map]
    
    if not pool_indices_list:
        print("Error: None of the provided entity URIs were found in the entity map.")
        return None
    
    pool_indices = np.array(pool_indices_list, dtype=np.int64)

    # 3. Load ONNX model and run inference
    ort_session = onnxruntime.InferenceSession(model_path)
    
    ort_inputs = {
        'x': x.numpy(),
        'edge_index': edge_index.numpy(),
        'edge_type': edge_type.numpy(),
        'pool_indices': pool_indices
    }
    
    ort_outs = ort_session.run(None, ort_inputs)
    pooled_embedding = ort_outs[0]
    
    # Squeeze the output to be a 1-D vector for compatibility with distance metrics
    return np.squeeze(pooled_embedding)

if __name__ == "__main__":
    out_dir = "out"
    model_path = os.path.join(out_dir, "embed_entities_neutral.onnx")
    data_path = os.path.join(out_dir, "embed_entities.pt")

    entities_to_pool = [
        "http://edugraph.io/edu#IntegerAddition",
        "http://edugraph.io/edu#IntegerSubtraction"
    ]

    print(f"Attempting to get pooled embedding for: {', '.join(e.split('#')[-1] for e in entities_to_pool)}")
    
    embedding = embed_entities(entities_to_pool, model_path, data_path)

    if embedding is not None:
        print("\nSuccessfully got pooled embedding.")
        print(f"Shape: {embedding.shape}")
        print(f"Embedding (first 5 elements): {embedding[0, :5]}")
