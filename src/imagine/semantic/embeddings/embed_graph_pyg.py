import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv
import numpy as np
from rdflib import Graph, URIRef, Literal
from collections import defaultdict
import itertools
import urllib.request


# --- 1. Data Loading: Load RDF from a URL ---

def load_kg_from_url(url):
    """
    Loads an RDF graph from a given URL.
    """
    g = Graph()
    print(f"Attempting to load RDF data from: {url}")
    try:
        g.parse(url, format="xml")
        print(f"Successfully loaded KG with {len(g)} triples.")
    except Exception as e:
        print(f"Failed to load or parse the graph. Error: {e}")
        return Graph()
    return g


# --- 2. Graph Preprocessing: Convert RDF to PyG Graph ---

def build_pyg_graph(rdf_graph):
    """
    Parses an rdflib.Graph and converts it into a PyG graph.
    """
    if len(rdf_graph) == 0:
        print("RDF graph is empty. Cannot build PyG graph.")
        return None, None, None, None

    all_nodes = set(rdf_graph.subjects()) | set(rdf_graph.objects())
    all_entities = sorted([node for node in all_nodes if isinstance(node, URIRef)])
    all_relations = sorted(list(set(rdf_graph.predicates())))

    entity_to_id = {entity: i for i, entity in enumerate(all_entities)}
    relation_to_id = {relation: i for i, relation in enumerate(all_relations)}
    id_to_entity = {i: entity for entity, i in entity_to_id.items()}

    src, rel, dst = [], [], []
    for s, p, o in rdf_graph:
        if s in entity_to_id and o in entity_to_id:
            src.append(entity_to_id[s])
            rel.append(relation_to_id[p])
            dst.append(entity_to_id[o])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type = torch.tensor(rel, dtype=torch.long)

    node_features = torch.randn(len(all_entities), 16)

    data = Data(x=node_features, edge_index=edge_index, edge_type=edge_type)
    data.num_nodes = len(all_entities)

    print(f"Built PyG graph with {data.num_nodes} nodes and {data.num_edges} edges.")
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


# --- Main Execution ---

if __name__ == "__main__":
    rdf_url = "https://github.com/christian-bick/edugraph-ontology/releases/download/v0.4.0/core-ontology.rdf"
    rdf_graph = load_kg_from_url(rdf_url)

    if len(rdf_graph) > 0:
        pyg_data, entity_map, rel_map, id_map = build_pyg_graph(rdf_graph)

        in_dim = pyg_data.x.shape[1]
        h_dim = 32
        out_dim = 32
        num_relations = len(rel_map)

        rgcn_model = RGCN(in_dim, h_dim, out_dim, num_relations)
        scorer_model = DistMultScorer(num_relations, out_dim)

        all_params = itertools.chain(rgcn_model.parameters(), scorer_model.parameters())
        optimizer = torch.optim.Adam(all_params, lr=0.01)

        trained_model = train(pyg_data, rgcn_model, scorer_model, optimizer, epochs=100)

        print("\n--- Generated Embeddings ---")
        trained_model.eval()
        with torch.no_grad():
            final_embeddings = trained_model(pyg_data.x, pyg_data.edge_index, pyg_data.edge_type)

        print(f"Shape of final embeddings tensor: {final_embeddings.shape}")

        target_entities = [
            "http://edugraph.io/edu#IntegerAddition",
            "http://edugraph.io/edu#IntegerSubtraction",
            "http://edugraph.io/edu#ProcedureExecution",
            "http://edugraph.io/edu#NumbersWithoutZero",
            "http://edugraph.io/edu#NumbersSmaller10"
        ]

        for entity_uri in target_entities:
            entity_ref = URIRef(entity_uri)
            if entity_ref in entity_map:
                entity_id = entity_map[entity_ref]
                embedding_vector = final_embeddings[entity_id]
                print(f"\nEmbedding for: {entity_uri.split('#')[-1]}")
                print(embedding_vector[:5].numpy())
            else:
                print(f"\nEntity not found in map: {entity_uri.split('#')[-1]}")