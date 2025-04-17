# Approximate Dynamic Programming (ADP) for Capacitated Vehicle Routing Problem (CVRP) with Residual Gated GCN

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import namedtuple

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

# -------------------- Training and Visualization --------------------
if __name__ == "__main__":
    # Parameters
    node_dim = 64
    num_layers = 3
    learning_rate = 0.001
    num_epochs = 200

    depot = Customer(id=0, x=50, y=50, demand=0)
    customers = [Customer(id=i, x=random.randint(0, 100), y=random.randint(0, 100), demand=random.randint(1, 10)) for i in range(1, 11)]
    vehicle_capacity = 20

    # Prepare graph inputs
    all_customers = [depot] + customers
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

    # Initialize model and optimizer
    model = ResidualGatedGCNModelVRP(node_dim=node_dim, num_layers=num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_history = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        logits, loss = model(node_features, adj_matrix, edge_weights)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.4f}")

    # Plot training loss
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.grid(True)
    plt.show()
