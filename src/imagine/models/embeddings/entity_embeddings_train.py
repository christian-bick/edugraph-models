import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv
import numpy as np
from rdflib import Graph, URIRef, Literal
from collections import defaultdict
import itertools

import os


from imagine.semantic.ontology_loader import load_ontology_rdflib

class RGCN(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, num_rels):
        super(RGCN, self).__init__()
        self.conv1 = RGCNConv(in_dim, h_dim, num_rels)
        self.conv2 = RGCNConv(h_dim, out_dim, num_rels)

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_type)
        return x

class RotatEScorer(nn.Module):
    def __init__(self, num_rels, embedding_dim):
        super(RotatEScorer, self).__init__()
        self.embedding_dim = embedding_dim

        # Relations are modeled as rotations, so they have a phase component.
        self.rel_embedding = nn.Embedding(num_rels, self.embedding_dim)

        # Initialize relation phases to be between 0 and 2*pi
        nn.init.uniform_(
            tensor=self.rel_embedding.weight,
            a=0,
            b=2 * np.pi
        )

    def forward(self, head_emb, tail_emb, rel_idx):
        # Separate real and imaginary parts
        head_real, head_imag = torch.chunk(head_emb, 2, dim=-1)
        tail_real, tail_imag = torch.chunk(tail_emb, 2, dim=-1)

        # Get relation phase
        relation_phase = self.rel_embedding(rel_idx)

        # Convert phase to complex number (re = cos(phase), im = sin(phase))
        rel_real = torch.cos(relation_phase)
        rel_imag = torch.sin(relation_phase)

        # Complex multiplication: (h_r + i*h_i) * (r_r + i*r_i) = (h_r*r_r - h_i*r_i) + i*(h_r*r_i + h_i*r_r)
        re_score = head_real * rel_real - head_imag * rel_imag
        im_score = head_real * rel_imag + head_imag * rel_real

        # The score is the L2 distance between the rotated head and the tail
        # score = -|| h*r - t ||
        score = torch.stack([re_score - tail_real, im_score - tail_imag], dim=0)
        score = score.norm(dim=0)

        # We return the negative distance, as higher scores should be better.
        return -score.sum(dim=-1)

class InferenceModel(nn.Module):
    def __init__(self, rgcn_model):
        super(InferenceModel, self).__init__()
        self.rgcn = rgcn_model
        self.rgcn.eval()

    def forward(self, x, edge_index, edge_type, pool_indices):
        node_embeddings = self.rgcn(x, edge_index, edge_type)

        # --- Weighted Pooling Logic ---
        selected_embeddings = node_embeddings.index_select(0, pool_indices)
        selected_features = x.index_select(0, pool_indices)

        # Extract one-hot encoded types (first 3 columns of features)
        type_features = selected_features[:, :3]

        # Define weights for Area, Scope, Ability
        weights_tensor = torch.tensor([4.0, 1.0, 2.0], device=x.device)

        # Calculate weight for each entity by dot product with one-hot vector
        entity_weights = torch.matmul(type_features, weights_tensor)

        # Perform weighted average: sum(weight_i * embedding_i) / sum(weight_i)
        # Add a small epsilon to the denominator to avoid division by zero
        sum_of_weights = torch.sum(entity_weights) + 1e-9

        # Reshape weights to (N, 1) to broadcast correctly with embeddings (N, D)
        weighted_embeddings = selected_embeddings * entity_weights.unsqueeze(1)
        
        pooled_embedding = torch.sum(weighted_embeddings, dim=0, keepdim=True) / sum_of_weights
        
        return pooled_embedding


# Define URIs for filtering
EDU_NS = "http://edugraph.io/edu#"
AREA_CLASS = URIRef(EDU_NS + "Area")
SCOPE_CLASS = URIRef(EDU_NS + "Scope")
ABILITY_CLASS = URIRef(EDU_NS + "Ability")
PART_OF_AREA_PRED = URIRef(EDU_NS + "partOfArea")
PART_OF_SCOPE_PRED = URIRef(EDU_NS + "partOfScope")
PART_OF_ABILITY_PRED = URIRef(EDU_NS + "partOfAbility")
RDF_TYPE = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")

def rdf_to_pyg_graph(rdf_graph):
    """
    Parses an rdflib.Graph and converts it into a PyG graph,
    filtering for specific entity types and relationships.
    """
    if len(rdf_graph) == 0:
        print("RDF graph is empty. Cannot build PyG graph.")
        return None, None, None, None

    # 1. Filter entities by type (Area, Scope, Ability)
    valid_entities = defaultdict(str) # Store entity -> type_string (e.g., "Area")
    for s, p, o in rdf_graph:
        if p == RDF_TYPE and o in [AREA_CLASS, SCOPE_CLASS, ABILITY_CLASS]:
            if isinstance(s, URIRef):
                if o == AREA_CLASS:
                    valid_entities[s] = "Area"
                elif o == SCOPE_CLASS:
                    valid_entities[s] = "Scope"
                elif o == ABILITY_CLASS:
                    valid_entities[s] = "Ability"

    if not valid_entities:
        print("No valid entities of type Area, Scope, or Ability found after filtering.")
        return None, None, None, None

    all_entities_list = sorted(list(valid_entities.keys()))
    entity_to_id = {entity: i for i, entity in enumerate(all_entities_list)}
    id_to_entity = {i: entity for entity, i in entity_to_id.items()}

    # 2. Filter relations based on allowed predicates and valid entities
    allowed_predicates = [PART_OF_AREA_PRED, PART_OF_SCOPE_PRED, PART_OF_ABILITY_PRED]
    # Ensure only these are considered relations for the graph
    all_relations = sorted(list(set(allowed_predicates)))
    relation_to_id = {relation: i for i, relation in enumerate(all_relations)}

    src, rel, dst = [], [], []
    for s, p, o in rdf_graph:
        if p in allowed_predicates and s in valid_entities and o in valid_entities:
            src.append(entity_to_id[s])
            rel.append(relation_to_id[p])
            dst.append(entity_to_id[o])

    if not src:
        print("No valid triples found after filtering entities and relations.")
        return None, None, None, None

    else:
        print(f"Found src={len(src)} rel={len(rel)} dst={len(rel)}")

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type = torch.tensor(rel, dtype=torch.long)

    # 3. Prepare node features with one-hot encoding for type
    num_entities = len(all_entities_list)
    type_one_hot = torch.zeros(num_entities, 3) # 3 for Area, Scope, Ability types
    random_features = torch.randn(num_entities, 61) # 61 random features + 3 one-hot features = 64

    for i, entity in enumerate(all_entities_list):
        entity_type = valid_entities[entity]
        if entity_type == "Area":
            type_one_hot[i, 0] = 1
        elif entity_type == "Scope":
            type_one_hot[i, 1] = 1
        elif entity_type == "Ability":
            type_one_hot[i, 2] = 1

    # Concatenate one-hot type features with random features
    node_features = torch.cat([type_one_hot, random_features], dim=1)

    data = Data(x=node_features, edge_index=edge_index, edge_type=edge_type)
    data.num_nodes = num_entities

    print(f"Built PyG graph with {data.num_nodes} nodes and {data.num_edges} edges after filtering.")
    print(f"Number of relation types: {len(all_relations)}")

    return data, entity_to_id, relation_to_id, id_to_entity

def train_rgcn(data, model, scorer, optimizer, epochs=50, neg_sample_size=32):
    """
    Main training loop for the R-GCN model using RotatE scoring
    with self-adversarial negative sampling.
    """
    print("\n--- Starting Model Training with RotatE Scorer ---")

    pos_heads, pos_rels, pos_tails = data.edge_index[0], data.edge_type, data.edge_index[1]

    for epoch in range(epochs):
        model.train()
        scorer.train()

        # Get all node embeddings (real and imaginary parts)
        node_embeddings = model(data.x, data.edge_index, data.edge_type)

        # Positive samples score
        pos_head_emb = node_embeddings[pos_heads]
        pos_tail_emb = node_embeddings[pos_tails]
        pos_score = scorer(pos_head_emb, pos_tail_emb, pos_rels)

        # Negative sampling
        num_pos_samples = len(pos_heads)
        
        # Repeat positive heads and relations for negative samples
        neg_head_emb = pos_head_emb.repeat(neg_sample_size, 1)
        neg_rel = pos_rels.repeat(neg_sample_size)

        # Generate random tails for negative samples
        corrupted_tails = torch.randint(0, data.num_nodes, (num_pos_samples * neg_sample_size,))
        neg_tail_emb = node_embeddings[corrupted_tails]

        neg_score = scorer(neg_head_emb, neg_tail_emb, neg_rel)

        # Self-adversarial loss calculation
        neg_score = neg_score.view(num_pos_samples, -1)
        pos_score = pos_score.view(num_pos_samples, -1)

        # The paper uses a temperature-scaled softmax for weighting negative samples
        softmax_weights = F.softmax(neg_score * 0.5, dim=1).detach()
        
        # Loss for negative samples
        neg_loss = torch.sum(softmax_weights * F.logsigmoid(-neg_score))

        # Loss for positive samples
        pos_loss = torch.sum(F.logsigmoid(pos_score))

        # Total loss is the negative log likelihood
        loss = -(pos_loss + neg_loss) / (2 * num_pos_samples)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:02d}/{epochs}, Loss: {loss.item():.4f}")

    print("--- Training Finished ---")
    return model

# --- Main Execution ---

def train_model_from_graph(rdf_graph):
    pyg_data, entity_map, rel_map, id_map = rdf_to_pyg_graph(rdf_graph)

    embedding_dim = 64 # This is the dimension of the complex embedding
    in_dim = pyg_data.x.shape[1]
    h_dim = 128 # Hidden dimension can be larger
    out_dim = embedding_dim * 2 # Real and imaginary parts
    num_relations = len(rel_map)

    rgcn_model = RGCN(in_dim, h_dim, out_dim, num_relations)
    scorer_model = RotatEScorer(num_relations, embedding_dim)

    all_params = itertools.chain(rgcn_model.parameters(), scorer_model.parameters())
    optimizer = torch.optim.Adam(all_params, lr=0.01)

    trained_model = train_rgcn(pyg_data, rgcn_model, scorer_model, optimizer, epochs=100)

    print("\n--- Create and Export ONNX Model ---")

    # Create the inference model
    inference_model = InferenceModel(trained_model)
    inference_model.eval()

    # Prepare for export
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    onnx_path = os.path.join(temp_dir, "embed_edugraph_labels.onnx")
    data_path = os.path.join(temp_dir, "embed_edugraph_labels.pt")

    # Dummy input for export.
    dummy_pool_indices = torch.tensor([0, 1], dtype=torch.long)

    # Export the model to ONNX
    print(f"Exporting model to {onnx_path}...")
    torch.onnx.export(
        inference_model,
        (pyg_data.x, pyg_data.edge_index, pyg_data.edge_type, dummy_pool_indices),
        onnx_path,
        input_names=['x', 'edge_index', 'edge_type', 'pool_indices'],
        output_names=['pooled_embedding'],
        dynamic_axes={
            'pool_indices': {0: 'num_to_pool'}
        },
        opset_version=12
    )
    print("Model exported successfully.")

    # Save necessary data for inference
    print(f"Saving inference data to {data_path}...")
    inference_data = {
        'x': pyg_data.x,
        'edge_index': pyg_data.edge_index,
        'edge_type': pyg_data.edge_type,
        'entity_map': {str(k): v for k, v in entity_map.items()},
    }
    torch.save(inference_data, data_path)
    print("Inference data saved.")

if __name__ == "__main__":
    rdf_url = "https://github.com/christian-bick/edugraph-ontology/releases/download/v0.4.0/core-ontology.rdf"
    ontology = load_ontology_rdflib(rdf_url)

    if ontology:
        train_model_from_graph(ontology)

