import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv
import numpy as np
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDFS
from collections import defaultdict
import itertools
import os
import re

from sentence_transformers import SentenceTransformer

from imagine.ontology_loader import load_ontology_rdflib

class RGCN(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, num_rels, dropout=0.5):
        super(RGCN, self).__init__()
        self.conv1 = RGCNConv(in_dim, h_dim, num_rels)
        self.conv2 = RGCNConv(h_dim, out_dim, num_rels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_type)
        return x

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

class InferenceModelBiased(nn.Module):
    def __init__(self, rgcn_model):
        super(InferenceModelBiased, self).__init__()
        self.rgcn = rgcn_model
        self.rgcn.eval()

    def forward(self, x, edge_index, edge_type, pool_indices, node_types):
        node_embeddings = self.rgcn(x, edge_index, edge_type)

        # --- Weighted Pooling Logic ---
        selected_embeddings = node_embeddings.index_select(0, pool_indices)
        
        # Get the types for the selected nodes
        selected_types = node_types[pool_indices]

        # Define weights for Area, Scope, Ability (0: Area, 1: Scope, 2: Ability)
        weights_map = torch.tensor([4.0, 1.0, 2.0], device=x.device)
        entity_weights = weights_map[selected_types]

        # Perform weighted average
        sum_of_weights = torch.sum(entity_weights) + 1e-9
        weighted_embeddings = selected_embeddings * entity_weights.unsqueeze(1)
        pooled_embedding = torch.sum(weighted_embeddings, dim=0, keepdim=True) / sum_of_weights
        
        return pooled_embedding

class InferenceModelNeutral(nn.Module):
    def __init__(self, rgcn_model):
        super(InferenceModelNeutral, self).__init__()
        self.rgcn = rgcn_model
        self.rgcn.eval()

    def forward(self, x, edge_index, edge_type, pool_indices):
        node_embeddings = self.rgcn(x, edge_index, edge_type)
        # --- Neutral Mean Pooling Logic ---
        pooled_embedding = torch.mean(node_embeddings.index_select(0, pool_indices), dim=0, keepdim=True)
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
    Parses an rdflib.Graph and converts it into a PyG graph.
    This version generates node features from text embeddings and includes node type information.
    """
    if len(rdf_graph) == 0:
        print("RDF graph is empty. Cannot build PyG graph.")
        return None, None, None, None

    # 1. Filter entities and store their types
    valid_entities = {}
    type_map = {"Area": 0, "Scope": 1, "Ability": 2}
    for s, p, o in rdf_graph.triples((None, RDF_TYPE, None)):
        if o in [AREA_CLASS, SCOPE_CLASS, ABILITY_CLASS] and isinstance(s, URIRef):
            if o == AREA_CLASS:
                valid_entities[s] = "Area"
            elif o == SCOPE_CLASS:
                valid_entities[s] = "Scope"
            elif o == ABILITY_CLASS:
                valid_entities[s] = "Ability"

    if not valid_entities:
        print("No valid entities of type Area, Scope, or Ability found.")
        return None, None, None, None

    all_entities_list = sorted(list(valid_entities.keys()))
    entity_to_id = {entity: i for i, entity in enumerate(all_entities_list)}
    
    # Create node type tensor
    node_types_list = [type_map[valid_entities[uri]] for uri in all_entities_list]
    node_types = torch.tensor(node_types_list, dtype=torch.long)

    # 2. Filter relations
    allowed_predicates = [PART_OF_AREA_PRED, PART_OF_SCOPE_PRED, PART_OF_ABILITY_PRED]
    all_relations = sorted(list(set(allowed_predicates)))
    relation_to_id = {relation: i for i, relation in enumerate(all_relations)}

    src, rel, dst = [], [], []
    for s, p, o in rdf_graph:
        if p in allowed_predicates and s in valid_entities and o in valid_entities:
            src.append(entity_to_id[s])
            rel.append(relation_to_id[p])
            dst.append(entity_to_id[o])

    if not src:
        print("No valid triples found after filtering.")
        return None, None, None, None

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type = torch.tensor(rel, dtype=torch.long)

    # 3. Prepare node features from text embeddings
    print("Generating node features from text definitions...")
    text_embedder = SentenceTransformer('all-MiniLM-L6-v2')
    DEFINITION_PREDICATE = RDFS.isDefinedBy
    LABEL_PREDICATE = RDFS.label

    text_for_embedding = []
    for entity_uri in all_entities_list:
        definition = next((str(o) for s, p, o in rdf_graph.triples((entity_uri, DEFINITION_PREDICATE, None))), None)
        if not definition:
            definition = next((str(o) for s, p, o in rdf_graph.triples((entity_uri, LABEL_PREDICATE, None))), None)
        if not definition:
            fragment = str(entity_uri).split('#')[-1]
            definition = ' '.join(re.findall('[A-Z][^A-Z]*', fragment))
        text_for_embedding.append(definition)

    node_features = torch.tensor(text_embedder.encode(text_for_embedding), dtype=torch.float)

    data = Data(x=node_features, edge_index=edge_index, edge_type=edge_type, node_types=node_types)
    data.num_nodes = len(all_entities_list)

    print(f"Built PyG graph with {data.num_nodes} nodes. Node feature dim: {data.x.shape[1]}")
    return data, entity_to_id, relation_to_id, {i: e for e, i in entity_to_id.items()}

def train_rgcn(data, model, scorer, optimizer, scheduler, epochs=50, margin=1.0):
    print("\n--- Starting Model Training with DistMult Scorer (Margin Loss) ---")
    pos_heads, pos_rels, pos_tails = data.edge_index[0], data.edge_type, data.edge_index[1]

    for epoch in range(epochs):
        model.train()
        scorer.train()
        node_embeddings = model(data.x, data.edge_index, data.edge_type)
        pos_scores = scorer(node_embeddings, pos_heads, pos_tails, pos_rels)
        num_pos_samples = len(pos_heads)
        corrupted_tails = torch.randint(0, data.num_nodes, (num_pos_samples,))
        neg_scores = scorer(node_embeddings, pos_heads, corrupted_tails, pos_rels)
        loss = torch.mean(F.relu(margin - (pos_scores - neg_scores)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:02d}/{epochs}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    print("--- Training Finished ---")
    return model

def train_model_from_graph(rdf_graph):
    pyg_data, entity_map, rel_map, id_map = rdf_to_pyg_graph(rdf_graph)

    embedding_dim = 128
    in_dim = pyg_data.x.shape[1]
    h_dim = 256
    out_dim = embedding_dim
    num_relations = len(rel_map)

    rgcn_model = RGCN(in_dim, h_dim, out_dim, num_relations, dropout=0.5)
    scorer_model = DistMultScorer(num_relations, embedding_dim)

    all_params = itertools.chain(rgcn_model.parameters(), scorer_model.parameters())
    optimizer = torch.optim.Adam(all_params, lr=0.01, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    trained_model = train_rgcn(pyg_data, rgcn_model, scorer_model, optimizer, scheduler, epochs=300)

    print("\n--- Create and Export ONNX Models (Biased and Neutral) ---")

    inference_model_biased = InferenceModelBiased(trained_model)
    inference_model_biased.eval()
    inference_model_neutral = InferenceModelNeutral(trained_model)
    inference_model_neutral.eval()

    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)

    onnx_path_biased = os.path.join(out_dir, "embed_entities_biased.onnx")
    onnx_path_neutral = os.path.join(out_dir, "embed_entities_neutral.onnx")
    data_path = os.path.join(out_dir, "embed_entities_text.pt")

    dummy_pool_indices = torch.tensor([0, 1], dtype=torch.long)

    print(f"Exporting BIASED model to {onnx_path_biased}...")
    torch.onnx.export(
        inference_model_biased,
        (pyg_data.x, pyg_data.edge_index, pyg_data.edge_type, dummy_pool_indices, pyg_data.node_types),
        onnx_path_biased,
        input_names=['x', 'edge_index', 'edge_type', 'pool_indices', 'node_types'],
        output_names=['pooled_embedding'],
        dynamic_axes={'pool_indices': {0: 'num_to_pool'}},
        opset_version=12
    )
    print("Biased model exported successfully.")

    print(f"Exporting NEUTRAL model to {onnx_path_neutral}...")
    torch.onnx.export(
        inference_model_neutral,
        (pyg_data.x, pyg_data.edge_index, pyg_data.edge_type, dummy_pool_indices),
        onnx_path_neutral,
        input_names=['x', 'edge_index', 'edge_type', 'pool_indices'],
        output_names=['pooled_embedding'],
        dynamic_axes={'pool_indices': {0: 'num_to_pool'}},
        opset_version=12
    )
    print("Neutral model exported successfully.")

    print(f"Saving inference data to {data_path}...")
    inference_data = {
        'x': pyg_data.x,
        'edge_index': pyg_data.edge_index,
        'edge_type': pyg_data.edge_type,
        'entity_map': {str(k): v for k, v in entity_map.items()},
        'node_types': pyg_data.node_types,
    }
    torch.save(inference_data, data_path)
    print("Inference data saved.")

if __name__ == "__main__":
    rdf_url = "https://github.com/christian-bick/edugraph-ontology/releases/download/v0.4.0/core-ontology.rdf"
    ontology = load_ontology_rdflib(rdf_url)

    if ontology:
        train_model_from_graph(ontology)
