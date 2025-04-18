import os
import json
import csv
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

    x_nodes_coord = torch.tensor(node_features, dtype=torch.float32, device=device).unsqueeze(0)
    x_nodes = torch.tensor([[0 if i == 0 else 1 for i in range(N)]], dtype=torch.long, device=device)
    x_edges = torch.zeros((1, N, N), dtype=torch.long, device=device)

    edge_values = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            if i != j:
                edge_values[i, j] = np.linalg.norm(coords[i] - coords[j])
    x_edges_values = torch.tensor(edge_values, dtype=torch.float32, device=device).unsqueeze(0)

    return x_edges, x_edges_values, x_nodes, x_nodes_coord

# -------------------- Validador de rutas --------------------
def is_valid_tour(tour, demands, vehicle_capacity):
    if tour[0] != 0 or tour[-1] != 0:
        return False
    visited = set()
    current_capacity = 0
    for i in range(1, len(tour)):
        node = tour[i]
        if node == 0:
            if current_capacity > vehicle_capacity:
                return False
            current_capacity = 0
        else:
            if node in visited:
                return False
            visited.add(node)
            current_capacity += demands[node]
    return visited == set(range(1, len(demands)))

# -------------------- Loss personalizada --------------------
def custom_loss(y_pred_edges, y_true_edges, tour, demands, vehicle_capacity, penalty_weight=1000.0):
    ce_loss = nn.CrossEntropyLoss()
    B, N, _, C = y_pred_edges.shape
    y_pred_flat = y_pred_edges.view(B * N * N, C)
    y_true_flat = y_true_edges.view(B * N * N)
    loss = ce_loss(y_pred_flat, y_true_flat)

    if not is_valid_tour(tour, demands, vehicle_capacity):
        penalty = torch.tensor(penalty_weight, dtype=loss.dtype, device=loss.device)
        loss += penalty
    return loss

# -------------------- Train Loop --------------------
def train_with_lkh_custom(model, folder_path, config_path, device='cpu', epochs=10, lr=1e-4, limit=None):
    config = load_config(config_path)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    vrp_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".vrp")])
    tour_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".tour")])
    data_pairs = list(zip(vrp_files, tour_files))
    if limit:
        data_pairs = data_pairs[:limit]

    metrics_log = []

    for epoch in range(epochs):
        total_loss = 0
        epoch_log = []

        for vrp_file, tour_file in tqdm(data_pairs, desc=f"Epoch {epoch+1}/{epochs}"):
            vrp_path = os.path.join(folder_path, vrp_file)
            tour_path = os.path.join(folder_path, tour_file)

            instance_name = f"{vrp_file}|{tour_file}"
            status = "success"
            loss_val = None

            try:
                node_coords, demands, tour = load_instance(vrp_path, tour_path)
                vehicle_capacity = config.get('vehicle_capacity', 1e6)

                labels = generate_labels_from_tour(tour, len(node_coords))
                y_edges = torch.tensor(labels, dtype=torch.long, device=device).unsqueeze(0)

                x_edges, x_edges_values, x_nodes, x_nodes_coord = prepare_inputs(node_coords, demands, config, device)

                model.train()
                y_pred_edges, _ = model(x_edges, x_edges_values, x_nodes, x_nodes_coord)

                loss = custom_loss(y_pred_edges, y_edges, tour, demands, vehicle_capacity)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                total_loss += loss_val

            except Exception as e:
                status = f"failed: {str(e)}"

            epoch_log.append({
                "epoch": epoch + 1,
                "instance": instance_name,
                "loss": loss_val,
                "status": status
            })

        avg_loss = total_loss / len(data_pairs)
        print(f"\n✅ Epoch {epoch+1}, Avg Loss: {avg_loss:.6f}")
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pt")

        # Save instance log per epoch
        log_file = f"epoch_{epoch+1}_log.csv"
        with open(log_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["epoch", "instance", "loss", "status"])
            writer.writeheader()
            writer.writerows(epoch_log)

        metrics_log.append({
            "epoch": epoch + 1,
            "average_loss": avg_loss,
        })

    # Save summary JSON
    with open("training_metrics_summary.json", 'w') as f:
        json.dump(metrics_log, f, indent=4)

# -------------------- Run Training --------------------
if __name__ == "__main__":
    folder = r"C:\\Users\\SAC\\Documents\\Freddy_Folder\\FAU\\Spring 2025\\Independent Study\\vrp_nazari100_validation_seed4321-lkh"
    config_path = "configs/vrp.json"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    raw_config = load_config(config_path)
    config = SimpleNamespace(**raw_config)
    model = ResidualGatedGCNModelVRP(config, torch.float32, torch.long)

    train_with_lkh_custom(model, folder, config_path, device=device, epochs=10, limit=5000)
