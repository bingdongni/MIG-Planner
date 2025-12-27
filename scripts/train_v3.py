import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np
import os
from data_engine import RoadNetworkManager
from model import MIGPlannerV3


def train_v3():
    # 刷新数据缓存
    data = RoadNetworkManager().load_graph_data(radius=1500, refresh=True)
    model = MIGPlannerV3(n_feats=2, hidden=128).to('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 准备专家图
    G = nx.DiGraph()
    edge_index = data.edge_index.numpy()
    edge_attr = data.edge_attr.numpy()
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i], edge_index[1, i]
        # 成本函数：距离 / (限速 * 权重)
        cost = edge_attr[i][0] / (edge_attr[i][3] + 0.1)
        G.add_edge(u, v, weight=cost)

    print("MIG-Planner v3.0 工业级训练启动...")
    model.train()
    device = next(model.parameters()).device
    data = data.to(device)

    for i in range(201):
        start, end = np.random.choice(data.num_nodes, 2, replace=False)
        try:
            path = nx.dijkstra_path(G, start, end, weight='weight')
            target = torch.zeros(data.num_nodes, 1).to(device)
            for node in path: target[node] = 1.0

            optimizer.zero_grad()
            logits = model(data)
            # 使用带 Logits 的损失函数，稳定性更高
            loss = F.binary_cross_entropy_with_logits(logits, target)
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print(f"Step {i:03d} | Loss: {loss.item():.4f}")
        except:
            continue

    torch.save(model.state_dict(), "mig_v3.pth")
    print("✅ 权重保存成功：mig_v3.pth")


if __name__ == "__main__":
    train_v3()