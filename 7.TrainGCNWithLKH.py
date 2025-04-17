import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from model import ResidualGatedGCNModelVRP
from types import SimpleNamespace
# -------------------- Load config --------------------
def load_config(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

# -------------------- Load .vrp + .tour --------------------
def load_instance(vrp_path, tour_path):
    with open(vrp_path, 'r') as f:
        lines = f.readlines()

    coords_section = lines.index("NODE_COORD_SECTION\n") + 1
    demand_section = lines.index("DEMAND_SECTION\n") + 1

    node_coords = []
    while not lines[coords_section].startswith("DEMAND_SECTION"):
        parts = lines[coords_section].strip().split()
        if len(parts) < 3:
            break
        node_coords.append((int(parts[0]), float(parts[1]), float(parts[2])))
        coords_section += 1

    demands = []
    while not lines[demand_section].startswith("DEPOT_SECTION"):
        parts = lines[demand_section].strip().split()
        if len(parts) < 2:
            break
        demands.append(int(parts[1]))
        demand_section += 1

    with open(tour_path, 'r') as f:
        tour_lines = f.readlines()
    start = tour_lines.index("TOUR_SECTION\n") + 1
    end = tour_lines.index("-1\n")
    tour = [int(line.strip()) - 1 for line in tour_lines[start:end]]  # 0-indexed

    return node_coords, demands, tour

# -------------------- Generate Edge Labels --------------------
def generate_labels_from_tour(tour, num_nodes):
    label = np.zeros((num_nodes, num_nodes), dtype=np.int64)
    for i in range(len(tour) - 1):
        u, v = tour[i], tour[i+1]
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            label[u, v] = 1
    return label


# -------------------- Prepare Model Inputs --------------------
def prepare_inputs(node_coords, demands, config, device):
    coords = np.array([[x, y] for _, x, y in node_coords], dtype=np.float32)
    dmds = np.array(demands, dtype=np.float32).reshape(-1, 1)
    node_features = np.hstack([coords, dmds])
    N = len(node_coords)

    # x_nodes_coord: (1, N, 3)
    x_nodes_coord = torch.tensor(node_features, dtype=torch.float32, device=device).unsqueeze(0)

    # x_nodes: (1, N) — use dummy index 0 for depot, 1 for customer
    x_nodes = torch.tensor([[0 if i == 0 else 1 for i in range(N)]], dtype=torch.long, device=device)

    # x_edges: (1, N, N) — dummy type 0 for all
    x_edges = torch.zeros((1, N, N), dtype=torch.long, device=device)

    # x_edges_values: (1, N, N) — Euclidean distance
    edge_values = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            if i != j:
                edge_values[i, j] = np.linalg.norm(coords[i] - coords[j])
    x_edges_values = torch.tensor(edge_values, dtype=torch.float32, device=device).unsqueeze(0)

    return x_edges, x_edges_values, x_nodes, x_nodes_coord

# -------------------- Train Loop --------------------
def train_with_lkh_custom(model, folder_path, config_path, device='cpu', epochs=10, lr=1e-4, limit=None):
    config = load_config(config_path)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    edge_cw = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)  # class weights for [0, 1]

    vrp_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".vrp")])
    tour_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".tour")])
    data_pairs = list(zip(vrp_files, tour_files))
    if limit:
        data_pairs = data_pairs[:limit]

    for epoch in range(epochs):
        total_loss = 0
        for vrp_file, tour_file in tqdm(data_pairs, desc=f"Epoch {epoch+1}/{epochs}"):
            vrp_path = os.path.join(folder_path, vrp_file)
            tour_path = os.path.join(folder_path, tour_file)

            node_coords, demands, tour = load_instance(vrp_path, tour_path)
            labels = generate_labels_from_tour(tour, len(node_coords))
            y_edges = torch.tensor(labels, dtype=torch.long, device=device).unsqueeze(0)

            x_edges, x_edges_values, x_nodes, x_nodes_coord = prepare_inputs(node_coords, demands, config, device)

            model.train()
            y_pred_edges, loss = model(x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges=y_edges, edge_cw=edge_cw)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data_pairs)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}")

        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pt")

# -------------------- Run Training --------------------
if __name__ == "__main__":
    folder = r"C:\\Users\\SAC\\Documents\\Freddy_Folder\\FAU\\Spring 2025\\Independent Study\\vrp_nazari100_validation_seed4321-lkh"
    config_path = "configs/vrp.json"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    raw_config = load_config(config_path)
    config = SimpleNamespace(**raw_config)
    model = ResidualGatedGCNModelVRP(config, torch.float32, torch.long)

    train_with_lkh_custom(model, folder, config_path, device=device, epochs=10, limit=5000)
