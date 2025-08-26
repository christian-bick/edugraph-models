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

# This new approach requires the sentence-transformers library
# Please install it: pip install sentence-transformers
from sentence_transformers import SentenceTransformer

from imagine.ontology_loader import load_ontology_rdflib

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

    def forward(self, x, edge_index, edge_type, pool_indices):
        node_embeddings = self.rgcn(x, edge_index, edge_type)

        # --- Weighted Pooling Logic ---
        selected_embeddings = node_embeddings.index_select(0, pool_indices)
        
        # The input features `x` are now text embeddings, not one-hot vectors.
        # We cannot determine the type from `x` anymore for weighting.
        # For this text-based model, the biased pooling is effectively disabled
        # and will behave like the neutral one. A more advanced implementation
        # would pass type information separately.
        pooled_embedding = torch.mean(selected_embeddings, dim=0, keepdim=True)
        
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
    Parses an rdflib.Graph and converts it into a PyG graph,
    filtering for specific entity types and relationships.
    MODIFIED: This version generates node features from text embeddings.
    """
    if len(rdf_graph) == 0:
        print("RDF graph is empty. Cannot build PyG graph.")
        return None, None, None, None

    # 1. Filter entities by type (Area, Scope, Ability)
    valid_entities = defaultdict(str)
    for s, p, o in rdf_graph.triples((None, RDF_TYPE, None)):
        if o in [AREA_CLASS, SCOPE_CLASS, ABILITY_CLASS] and isinstance(s, URIRef):
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

    # 2. Filter relations based on allowed predicates
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
        print("No valid triples found after filtering entities and relations.")
        return None, None, None, None

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type = torch.tensor(rel, dtype=torch.long)

    # 3. Prepare node features using text embeddings of definitions
    print("Generating node features from text definitions using sentence-transformers...")
    text_embedder = SentenceTransformer('all-MiniLM-L6-v2')

    text_for_embedding = []
    # Using rdfs:isDefinedBy as found in the test ontology file
    DEFINITION_PREDICATE = RDFS.isDefinedBy
    LABEL_PREDICATE = RDFS.label

    for entity_uri in all_entities_list:
        # Try to get definition
        definition = next((str(o) for s, p, o in rdf_graph.triples((entity_uri, DEFINITION_PREDICATE, None))), None)
        
        # If no definition, fall back to the label
        if not definition:
            definition = next((str(o) for s, p, o in rdf_graph.triples((entity_uri, LABEL_PREDICATE, None))), None)
        
        # If still no text, use the URI fragment, converted from CamelCase
        if not definition:
            fragment = str(entity_uri).split('#')[-1]
            definition = ' '.join(re.findall('[A-Z][^A-Z]*', fragment))
        
        text_for_embedding.append(definition)

    node_features = torch.tensor(text_embedder.encode(text_for_embedding), dtype=torch.float)

    data = Data(x=node_features, edge_index=edge_index, edge_type=edge_type)
    data.num_nodes = len(all_entities_list)

    print(f"Built PyG graph with {data.num_nodes} nodes and {data.num_edges} edges.")
    print(f"Node feature dimension: {data.x.shape[1]}")

    return data, entity_to_id, relation_to_id, {i: e for e, i in entity_to_id.items()}

def train_rgcn(data, model, scorer, optimizer, epochs=50, margin=1.0):
    """
    Main training loop for the R-GCN model using DistMult scoring
    with a margin-based ranking loss.
    """
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

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:02d}/{epochs}, Loss: {loss.item():.4f}")

    print("--- Training Finished ---")
    return model

# --- Main Execution ---

def train_model_from_graph(rdf_graph):
    pyg_data, entity_map, rel_map, id_map = rdf_to_pyg_graph(rdf_graph)

    # Dimensions are now based on the sentence-transformer model
    embedding_dim = 128  # The dimension of the final graph embedding
    in_dim = pyg_data.x.shape[1] # Should be 384 for all-MiniLM-L6-v2
    h_dim = 256 # Hidden dimension
    out_dim = embedding_dim
    num_relations = len(rel_map)

    rgcn_model = RGCN(in_dim, h_dim, out_dim, num_relations)
    scorer_model = DistMultScorer(num_relations, embedding_dim)

    all_params = itertools.chain(rgcn_model.parameters(), scorer_model.parameters())
    optimizer = torch.optim.Adam(all_params, lr=0.001)

    trained_model = train_rgcn(pyg_data, rgcn_model, scorer_model, optimizer, epochs=300)

    print("\n--- Create and Export ONNX Models (Biased and Neutral) ---")

    # NOTE: With text-based features, the biased pooling is no longer possible
    # as the type information is not explicitly in the feature vector.
    # Both models will behave identically (neutral pooling).
    inference_model_biased = InferenceModelBiased(trained_model)
    inference_model_biased.eval()

    inference_model_neutral = InferenceModelNeutral(trained_model)
    inference_model_neutral.eval()

    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)

    onnx_path_biased = os.path.join(out_dir, "embed_entities_text_biased.onnx")
    onnx_path_neutral = os.path.join(out_dir, "embed_entities_text_neutral.onnx")
    data_path = os.path.join(out_dir, "embed_entities_text.pt")

    dummy_pool_indices = torch.tensor([0, 1], dtype=torch.long)

    print(f"Exporting BIASED model to {onnx_path_biased}...")
    torch.onnx.export(
        inference_model_biased,
        (pyg_data.x, pyg_data.edge_index, pyg_data.edge_type, dummy_pool_indices),
        onnx_path_biased,
        input_names=['x', 'edge_index', 'edge_type', 'pool_indices'],
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
    }
    torch.save(inference_data, data_path)
    print("Inference data saved.")

if __name__ == "__main__":
    rdf_url = "https://github.com/christian-bick/edugraph-ontology/releases/download/v0.4.0/core-ontology.rdf"
    ontology = load_ontology_rdflib(rdf_url)

    if ontology:
        train_model_from_graph(ontology)
