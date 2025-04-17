# Approximate Dynamic Programming (ADP) for Capacitated Vehicle Routing Problem (CVRP) with Residual Gated GCN and Beam Search

import numpy as np
import random
from collections import namedtuple, defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import heapq

# Define a Customer and Vehicle
Customer = namedtuple('Customer', ['id', 'x', 'y', 'demand'])

# -------------------- Problem Setup --------------------
class CVRPEnvironment:
    def __init__(self, depot, customers, vehicle_capacity):
        self.depot = depot
        self.customers = customers
        self.vehicle_capacity = vehicle_capacity

    def distance(self, c1, c2):
        return np.hypot(c1.x - c2.x, c1.y - c2.y)

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

# -------------------- Beam Search Policy --------------------
class PartialSolution:
    def __init__(self, route, cost, current, capacity, visited):
        self.route = route
        self.cost = cost
        self.current = current
        self.capacity = capacity
        self.visited = visited

    def __lt__(self, other):
        return self.cost < other.cost

    def state_key(self):
        return (tuple(sorted(self.visited)), self.current, self.capacity)

    def dominates(self, other):
        return (self.visited == other.visited and self.current == other.current
                and self.cost <= other.cost and self.capacity >= other.capacity
                and (self.cost < other.cost or self.capacity > other.capacity))

# -------------------- Beam Search Implementation --------------------
def beam_search(env, model, beam_width=5):
    all_customers = [env.depot] + env.customers
    N = len(all_customers)
    node_features = torch.tensor([[c.x, c.y, c.demand] for c in all_customers], dtype=torch.float32)
    edge_weights = torch.zeros((N, N, 1))
    adj_matrix = torch.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i != j:
                dist = env.distance(all_customers[i], all_customers[j])
                edge_weights[i, j, 0] = dist
                adj_matrix[i, j] = 1

    model.eval()
    with torch.no_grad():
        logits, _ = model(node_features, adj_matrix, edge_weights)
        heat = logits.numpy()

    beam = [PartialSolution(route=[0], cost=0, current=0, capacity=env.vehicle_capacity, visited=set())]
    visited_states = defaultdict(list)

    for _ in range(N * 2):  # allow via-depot moves
        next_beam = []
        for sol in beam:
            for j, cust in enumerate(env.customers, 1):
                if j in sol.visited:
                    continue
                demand = cust.demand
                # direct move
                if demand <= sol.capacity:
                    cost = env.distance(all_customers[sol.current], cust)
                    new_cost = sol.cost + cost
                    new_capacity = sol.capacity - demand
                    new_route = sol.route + [j]
                    new_visited = sol.visited | {j}
                    next_sol = PartialSolution(new_route, new_cost, j, new_capacity, new_visited)
                    next_beam.append(next_sol)
                else:
                    # via depot
                    cost = (env.distance(all_customers[sol.current], env.depot) +
                            env.distance(env.depot, cust))
                    new_cost = sol.cost + cost
                    new_capacity = env.vehicle_capacity - demand
                    new_route = sol.route + [0, j]
                    new_visited = sol.visited | {j}
                    next_sol = PartialSolution(new_route, new_cost, j, new_capacity, new_visited)
                    next_beam.append(next_sol)
        # Eliminate dominated solutions
        filtered = {}
        for sol in next_beam:
            key = sol.state_key()
            if key not in filtered or sol.dominates(filtered[key]):
                filtered[key] = sol
        beam = heapq.nsmallest(beam_width, filtered.values())
        if all(len(sol.visited) == len(env.customers) for sol in beam):
            break

    best = min(beam, key=lambda s: s.cost)
    return best

# -------------------- Example Usage --------------------
if __name__ == "__main__":
    depot = Customer(id=0, x=50, y=50, demand=0)
    customers = [Customer(id=i, x=random.randint(0, 100), y=random.randint(0, 100), demand=random.randint(1, 10)) for i in range(1, 6)]
    vehicle_capacity = 20

    env = CVRPEnvironment(depot, customers, vehicle_capacity)
    model = ResidualGatedGCNModelVRP()

    solution = beam_search(env, model, beam_width=5)
    print("\nBeam Search Route:", solution.route)
    print("Total Cost:", round(solution.cost, 2))
