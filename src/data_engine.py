import osmnx as ox
import networkx as nx
import torch
import os
import logging
from torch_geometric.utils import from_networkx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MIG.DataEngine_v3")


class RoadNetworkManager:
    def __init__(self, cache_dir: str = "data/processed"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.ref_point = (31.2335, 121.4844)

    def load_graph_data(self, radius: int = 1500, refresh: bool = False):
        cache_path = os.path.join(self.cache_dir, f"map_v3_r{radius}.pt")
        if os.path.exists(cache_path) and not refresh:
            return torch.load(cache_path, weights_only=False)

        logger.info(f"Querying Industrial OSM Data (Radius: {radius}m)...")
        raw_graph = ox.graph_from_point(self.ref_point, dist=radius, network_type='drive', simplify=True)
        graph = ox.truncate.largest_component(raw_graph, strongly=True)
        graph = nx.convert_node_labels_to_integers(graph)

        # 1. 节点属性对齐
        for node, data in graph.nodes(data=True):
            x, y = data.get('x', 0.0), data.get('y', 0.0)
            graph.nodes[node]['x_feat'] = [float(x) / 180.0, float(y) / 90.0]
            # 清理多余属性避免冲突
            keys = list(data.keys())
            for k in keys:
                if k not in ['x_feat']: del graph.nodes[node][k]

        # 2. 边属性对齐 (工业级清洗逻辑)
        for u, v, k, data in graph.edges(keys=True, data=True):
            length = float(data.get('length', 100.0)) / 1000.0
            oneway = 1.0 if data.get('oneway', False) else 0.0

            # 安全提取车道数
            lanes = data.get('lanes', 1)
            if isinstance(lanes, list): lanes = lanes[0]
            try:
                lanes = int(str(lanes)[0]) / 5.0
            except:
                lanes = 0.2

            # 安全提取限速
            maxspeed = data.get('maxspeed', 40)
            if isinstance(maxspeed, list): maxspeed = maxspeed[0]
            try:
                maxspeed = float(str(maxspeed).replace('km/h', '')) / 120.0
            except:
                maxspeed = 0.33

            # 强制注入统一名称的边特征向量
            graph.edges[u, v, k]['e_feat'] = [length, oneway, lanes, maxspeed]

            # 清理边上的原始属性，防止 from_networkx 扫描到不一致的 key
            original_keys = list(data.keys())
            for key in original_keys:
                if key != 'e_feat': del graph.edges[u, v, k][key]

        # 3. 转换并序列化
        pyg_graph = from_networkx(graph, group_node_attrs=['x_feat'], group_edge_attrs=['e_feat'])
        # 重命名边属性为标准的 edge_attr
        pyg_graph.edge_attr = pyg_graph.edge_attr.float()
        pyg_graph.x = pyg_graph.x.float()

        torch.save(pyg_graph, cache_path)
        logger.info(f"V3 Graph Cleansed & Loaded. Nodes: {pyg_graph.num_nodes}, Edges: {pyg_graph.num_edges}")
        return pyg_graph