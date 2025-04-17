import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import ResidualGatedGCNModelVRP
import json

# -------------------- Load config --------------------
def load_config(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

# -------------------- Load .vrp --------------------
def load_vrp_instance(vrp_path):
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

    return node_coords, demands

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

    return x_edges, x_edges_values, x_nodes, x_nodes_coord, coords

# -------------------- Plot Heatmap --------------------
def plot_heatmap(heat, coords, title="Predicted Heatmap"):
    fig, ax = plt.subplots(figsize=(8, 8))
    N = len(coords)
    coords = np.array(coords)

    for i in range(N):
        for j in range(N):
            if i != j and heat[i, j] > 0.1:
                ax.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]],
                        alpha=heat[i, j], color='r', linewidth=1.5 * heat[i, j])

    ax.scatter(coords[:, 0], coords[:, 1], c='blue', s=30)
    ax.scatter(coords[0, 0], coords[0, 1], c='green', s=80, label='Depot')
    ax.set_title(title)
    ax.legend()
    plt.grid(True)
    plt.show()

# -------------------- Evaluation Script --------------------
def evaluate_model(model_path, vrp_path, config_path, device='cpu'):
    config = load_config(config_path)
    model = ResidualGatedGCNModelVRP(config, torch.float32, torch.long)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    node_coords, demands = load_vrp_instance(vrp_path)
    x_edges, x_edges_values, x_nodes, x_nodes_coord, coords = prepare_inputs(node_coords, demands, config, device)

    with torch.no_grad():
        y_pred_edges, _ = model(x_edges, x_edges_values, x_nodes, x_nodes_coord)
        heat = torch.softmax(y_pred_edges[0], dim=-1)[:, :, 1].cpu().numpy()  # heat = P(edge_class == 1)

    plot_heatmap(heat, coords, title=f"Heatmap for {os.path.basename(vrp_path)}")

# -------------------- Run --------------------
if __name__ == "__main__":
    model_path = "model_epoch_10.pt"
    vrp_path = r"C:\\Users\\SAC\\Documents\\Freddy_Folder\\FAU\\Spring 2025\\Independent Study\\vrp_nazari100_validation_seed4321-lkh\\0001.lkh1.vrp"
    config_path = "configs/vrp.json"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    evaluate_model(model_path, vrp_path, config_path, device=device)
