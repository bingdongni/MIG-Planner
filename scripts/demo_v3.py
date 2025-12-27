import gradio as gr
import folium
from folium import plugins
import torch
import osmnx as ox
import networkx as nx
import os
import numpy as np
import io
from data_engine import RoadNetworkManager
from model import MIGPlannerV3
from live_traffic import AmapLiveBridge

AMAP_KEY = "12e626d096ffb40e92b677833b541b78"
MODEL_WEIGHTS = "mig_v3.pth"


class MIGDigitalTwinTerminal:
    def __init__(self, key):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.manager = RoadNetworkManager()
        self.bridge = AmapLiveBridge(key)

        # 1. 加载数据：使用训练时一致的 1500m 半径
        self.data = self.manager.load_graph_data(radius=1500).to(self.device)

        # 2. 初始化模型：hidden=128 必须与 train_v3.py 保持一致
        self.model = MIGPlannerV3(n_feats=2, hidden=128).to(self.device)

        # 3. 安全加载权重
        if os.path.exists(MODEL_WEIGHTS):
            self.model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=self.device, weights_only=False))
            print(f"✅ 成功挂载 v3.0 工业级权重: {MODEL_WEIGHTS}")
        else:
            print(f"⚠️ 警告: 未找到 {MODEL_WEIGHTS}, 将使用随机参数进行演示")
        self.model.eval()

        # 4. 构建底图拓扑：用于 Folium 渲染坐标映射
        raw_graph = ox.graph_from_point((31.2335, 121.4844), dist=1500, network_type='drive')
        self.G_nx = nx.convert_node_labels_to_integers(ox.truncate.largest_component(raw_graph, strongly=True))

    def update_digital_twin(self, s_ids, e_ids, live_on):
        """
        执行实况感知与群体路径空间推理
        """
        current_data = self.data.clone()
        temp_G = self.G_nx.copy()

        # 5. 接入实时高德交通流同步
        if live_on:
            traffic_info = self.bridge.get_live_weights(121.4844, 31.2335)
            edge_index = current_data.edge_index
            for i in range(edge_index.shape[1]):
                u, v = edge_index[0, i].item(), edge_index[1, i].item()
                # 语义匹配：将 API 数据映射至路网拓扑
                road_data = self.G_nx.get_edge_data(u, v)
                if road_data:
                    name = road_data[0].get('name', 'Unknown')
                    if isinstance(name, list): name = name[0]
                    # 更新张量特征以影响注意力分布
                    penalty = traffic_info.get(name, 1.0)
                    current_data.edge_attr[i, 0] *= penalty
                    temp_G[u][v][0]['length'] *= penalty

        # 6. 执行自适应 GAT 推理
        with torch.no_grad():
            logits = self.model(current_data)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

        # 7. 构建交互式数字孪生画布
        m = folium.Map(location=[31.2335, 121.4844], zoom_start=15, tiles='CartoDB dark_matter')

        heat_data = [[self.G_nx.nodes[i]['y'], self.G_nx.nodes[i]['x'], float(probs[i])]
                     for i in range(len(probs)) if i in self.G_nx.nodes]
        plugins.HeatMap(heat_data, radius=18, blur=12, min_opacity=0.4).add_to(m)

        try:
            starts = [int(x.strip()) for x in s_ids.split(',') if x.strip()]
            ends = [int(x.strip()) for x in e_ids.split(',') if x.strip()]
            for s, e in zip(starts, ends):
                if s in self.G_nx.nodes and e in self.G_nx.nodes:
                    path = nx.dijkstra_path(temp_G, s, e, weight='length')
                    path_coords = [[self.G_nx.nodes[n]['y'], self.G_nx.nodes[n]['x']] for n in path]
                    # 注入霓虹流光线
                    plugins.AntPath(path_coords, color='#00ff88', delay=800, weight=6).add_to(m)
                    folium.CircleMarker(path_coords[0], radius=6, color='cyan', fill=True).add_to(m)
                    folium.CircleMarker(path_coords[-1], radius=6, color='magenta', fill=True).add_to(m)
        except Exception as e:
            print(f"Routing Error: {e}")

        return m._repr_html_()


def run_v3_terminal():
    terminal = MIGDigitalTwinTerminal(AMAP_KEY)

    with gr.Blocks(title="MIG-Planner v3.0", theme=gr.themes.Soft()) as app:
        gr.HTML("""
            <div style="text-align:center; background: linear-gradient(135deg, #1a2a6c, #b21f1f, #fdbb2d); padding: 15px; border-radius: 8px;">
                <h1 style="color:white; margin:0;">MIG-Planner v3.0 | 城市级数字孪生决策系统</h1>
                <p style="color:white; opacity:0.8;">Industrial Spatio-Temporal Intelligent Engine</p>
            </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                s_in = gr.Textbox(label="起点集群 IDs", value="15, 30, 60")
                e_in = gr.Textbox(label="终点集群 IDs", value="110, 155, 190")
                live = gr.Checkbox(label="接入高德实时交通流 (Active)", value=True)
                btn = gr.Button("启动全城自适应推理", variant="primary")
                gr.Markdown("---")
                gr.Markdown(
                    "### 🛠 核心引擎状态\n- **架构**: Hierarchical Masked-GAT\n- **数据**: Live Amap REST API\n- **渲染**: Folium 3D Digital Twin")

            with gr.Column(scale=4):
                map_out = gr.HTML(value="<div style='height:600px; background:#111; border-radius:10px;'></div>")

        btn.click(terminal.update_digital_twin, [s_in, e_in, live], map_out)

    app.launch(server_name="127.0.0.1", inbrowser=True)


if __name__ == "__main__":
    run_v3_terminal()