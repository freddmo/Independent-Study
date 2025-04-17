# Approximate Dynamic Programming (ADP) for Capacitated Vehicle Routing Problem (CVRP) with Residual Gated GCN

import numpy as np
import random
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a Customer and Vehicle
Customer = namedtuple('Customer', ['id', 'x', 'y', 'demand'])

# -------------------- Problem Setup --------------------
class CVRPEnvironment:
    def __init__(self, depot, customers, vehicle_capacity):
        self.depot = depot
        self.customers = customers
        self.vehicle_capacity = vehicle_capacity
        self.reset()

    def reset(self):
        self.remaining_customers = {c.id: c for c in self.customers}
        self.vehicle_position = self.depot
        self.remaining_capacity = self.vehicle_capacity
        self.route = [self.depot.id]
        self.total_cost = 0

    def distance(self, c1, c2):
        return np.hypot(c1.x - c2.x, c1.y - c2.y)

    def available_actions(self):
        return [c for c in self.remaining_customers.values() if c.demand <= self.remaining_capacity]

    def step(self, action_customer):
        cost = self.distance(self.vehicle_position, action_customer)
        self.total_cost += cost
        self.remaining_capacity -= action_customer.demand
        self.route.append(action_customer.id)
        self.vehicle_position = action_customer
        del self.remaining_customers[action_customer.id]

        if not self.available_actions():
            self.total_cost += self.distance(self.vehicle_position, self.depot)
            self.vehicle_position = self.depot
            self.remaining_capacity = self.vehicle_capacity
            self.route.append(self.depot.id)

    def is_done(self):
        return len(self.remaining_customers) == 0

# -------------------- Residual Gated GCN Model --------------------
class ResidualGatedGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear_u = nn.Linear(in_dim, out_dim)
        self.linear_v = nn.Linear(in_dim, out_dim)
        self.gate = nn.Linear(in_dim * 2, out_dim)

    def forward(self, h, adj):
        u = self.linear_u(h)
        v = self.linear_v(h)
        out = torch.matmul(adj, v)
        gated = torch.sigmoid(self.gate(torch.cat([u, out], dim=-1)))
        return F.relu(h + gated * out)

class ResidualGatedGCNModelVRP(nn.Module):
    def __init__(self, node_dim=128, edge_dim=128, num_layers=3):
        super().__init__()
        self.node_embed = nn.Linear(3, node_dim)  # x, y, demand
        self.edge_embed = nn.Linear(1, edge_dim)  # distance

        self.gcn_layers = nn.ModuleList([
            ResidualGatedGCNLayer(node_dim, node_dim) for _ in range(num_layers)
        ])

        self.edge_classifier = nn.Sequential(
            nn.Linear(node_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, node_features, adj_matrix, edge_weights):
        h = self.node_embed(node_features)
        edge_embed = self.edge_embed(edge_weights)

        for gcn in self.gcn_layers:
            h = gcn(h, adj_matrix)

        N = h.size(0)
        h_i = h.unsqueeze(1).expand(-1, N, -1)
        h_j = h.unsqueeze(0).expand(N, -1, -1)
        edge_input = torch.cat([h_i, h_j], dim=-1)

        logits = self.edge_classifier(edge_input).squeeze(-1)
        loss = F.mse_loss(logits * adj_matrix, edge_weights.squeeze(-1))
        return logits, loss

# -------------------- Greedy Policy --------------------
def greedy_policy(model, env, all_customers, device='cpu'):
    current_node = env.vehicle_position.id
    available = env.available_actions()
    if not available:
        return None

    customer_ids = [c.id for c in all_customers]
    node_features = torch.tensor([[c.x, c.y, c.demand] for c in all_customers], dtype=torch.float32).to(device)

    # Reconstruct adjacency and edge weights
    N = len(all_customers)
    edge_weights = torch.zeros((N, N, 1), dtype=torch.float32).to(device)
    adj_matrix = torch.zeros((N, N), dtype=torch.float32).to(device)
    for i in range(N):
        for j in range(N):
            if i != j:
                dist = np.hypot(all_customers[i].x - all_customers[j].x, all_customers[i].y - all_customers[j].y)
                edge_weights[i, j, 0] = dist
                adj_matrix[i, j] = 1

    model.eval()
    with torch.no_grad():
        logits, _ = model(node_features, adj_matrix, edge_weights)
        scores = logits[current_node]

    available_ids = [c.id for c in available]
    best_id = max(available_ids, key=lambda i: scores[i].item() * -1)  # minimize distance or cost
    return next(c for c in available if c.id == best_id)

# -------------------- Example Usage --------------------
if __name__ == "__main__":
    depot = Customer(id=0, x=50, y=50, demand=0)
    customers = [Customer(id=i, x=random.randint(0, 100), y=random.randint(0, 100), demand=random.randint(1, 10)) for i in range(1, 6)]
    vehicle_capacity = 20

    env = CVRPEnvironment(depot, customers, vehicle_capacity)
    all_customers = [depot] + customers

    model = ResidualGatedGCNModelVRP()
    
    # Dummy train: forward once to simulate learning loss
    N = len(all_customers)
    node_features = torch.tensor([[c.x, c.y, c.demand] for c in all_customers], dtype=torch.float32)
    edge_weights = torch.zeros((N, N, 1))
    adj_matrix = torch.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                dist = np.hypot(all_customers[i].x - all_customers[j].x, all_customers[i].y - all_customers[j].y)
                edge_weights[i, j, 0] = dist
                adj_matrix[i, j] = 1

    logits, loss = model(node_features, adj_matrix, edge_weights)
    print("Loss (simulated training):", loss.item())

    # ----------------- Simulate Routing -----------------
    print("\nSimulating Greedy Route:")
    while not env.is_done():
        action = greedy_policy(model, env, all_customers)
        if action:
            env.step(action)
        else:
            # No valid actions: return to depot
            env.step(depot)

    print("Route Taken:", env.route)
    print("Total Cost:", env.total_cost)
