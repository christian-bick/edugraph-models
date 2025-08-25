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


# --- 2. Graph Preprocessing: Convert RDF to PyG Graph ---

# Define URIs for filtering
EDU_NS = "http://edugraph.io/edu#"
AREA_CLASS = URIRef(EDU_NS + "Area")
SCOPE_CLASS = URIRef(EDU_NS + "Scope")
ABILITY_CLASS = URIRef(EDU_NS + "Ability")
PART_OF_AREA_PRED = URIRef(EDU_NS + "partOfArea")
PART_OF_SCOPE_PRED = URIRef(EDU_NS + "partOfScope")
PART_OF_ABILITY_PRED = URIRef(EDU_NS + "partOfAbility")
RDF_TYPE = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")

def build_pyg_graph(rdf_graph):
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
    random_features = torch.randn(num_entities, 16) # Keep original random features

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


# --- 3. R-GCN Model Definition ---

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


# --- 4. Link Prediction Scorer ---

class DistMultScorer(nn.Module):
    def __init__(self, num_rels, embedding_dim):
        super(DistMultScorer, self).__init__()
        self.rel_embedding = nn.Embedding(num_rels, embedding_dim)

    def forward(self, node_embeddings, s, o, r):
        s_emb = node_embeddings[s]
        o_emb = node_embeddings[o]
        r_emb = self.rel_embedding(r)
        score = torch.sum(s_emb * r_emb * o_emb, dim=-1)
        return score


# --- 5. Training Loop ---

def train(data, model, scorer, optimizer, epochs=50):
    """
    Main training loop for the R-GCN model using a link prediction task.
    """
    print("\n--- Starting Model Training ---")

    for epoch in range(epochs):
        model.train()
        scorer.train()

        embeddings = model(data.x, data.edge_index, data.edge_type)

        pos_scores = scorer(embeddings, data.edge_index[0], data.edge_index[1], data.edge_type)

        num_edges = data.num_edges
        num_nodes = data.num_nodes
        corrupted_tails = torch.randint(0, num_nodes, (num_edges,))

        neg_scores = scorer(embeddings, data.edge_index[0], corrupted_tails, data.edge_type)

        pos_labels = torch.ones_like(pos_scores)
        neg_labels = torch.zeros_like(neg_scores)

        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([pos_labels, neg_labels])

        loss = F.binary_cross_entropy_with_logits(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:02d}/{epochs}, Loss: {loss.item():.4f}")

    print("--- Training Finished ---")
    return model


# --- 6. Inference Model with Pooling ---

class InferenceModel(nn.Module):
    def __init__(self, rgcn_model):
        super(InferenceModel, self).__init__()
        self.rgcn = rgcn_model
        self.rgcn.eval()

    def forward(self, x, edge_index, edge_type, pool_indices):
        node_embeddings = self.rgcn(x, edge_index, edge_type)
        pooled_embedding = torch.mean(node_embeddings.index_select(0, pool_indices), dim=0, keepdim=True)
        return pooled_embedding


# --- Main Execution ---

if __name__ == "__main__":
    rdf_url = "https://github.com/christian-bick/edugraph-ontology/releases/download/v0.4.0/core-ontology.rdf"
    rdf_graph = load_ontology_rdflib(rdf_url)

    if len(rdf_graph) > 0:
        pyg_data, entity_map, rel_map, id_map = build_pyg_graph(rdf_graph)

        in_dim = pyg_data.x.shape[1]
        h_dim = 128
        out_dim = 128
        num_relations = len(rel_map)

        rgcn_model = RGCN(in_dim, h_dim, out_dim, num_relations)
        scorer_model = DistMultScorer(num_relations, out_dim)

        all_params = itertools.chain(rgcn_model.parameters(), scorer_model.parameters())
        optimizer = torch.optim.Adam(all_params, lr=0.01)

        trained_model = train(pyg_data, rgcn_model, scorer_model, optimizer, epochs=100)

        print("\n--- Create and Export ONNX Model ---")

        # 1. Create the inference model
        inference_model = InferenceModel(trained_model)
        inference_model.eval()

        # 2. Prepare for export
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        onnx_path = os.path.join(temp_dir, "graph_embedding_model.onnx")
        data_path = os.path.join(temp_dir, "inference_data.pt")

        # Dummy input for export.
        dummy_pool_indices = torch.tensor([0, 1], dtype=torch.long)

        # 3. Export the model to ONNX
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

        # 4. Save necessary data for inference
        print(f"Saving inference data to {data_path}...")
        inference_data = {
            'x': pyg_data.x,
            'edge_index': pyg_data.edge_index,
            'edge_type': pyg_data.edge_type,
            'entity_map': entity_map,
        }
        torch.save(inference_data, data_path)
        print("Inference data saved.")