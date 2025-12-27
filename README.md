# MIG-Planner v3.0: 城市级数字孪生与自适应空间智能规划引擎

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-ee4c2c.svg)](https://pytorch.org/)

## 🌟 项目简介
MIG-Planner 是一款基于 **Adaptive Graph Attention Networks (GAT)** 的工业级路径规划系统。它能够实时感知城市路网的物理约束（单行道、限速、车道数），并动态集成 **高德地图 (Amap) 实时交通流** 数据，实现从实验室模拟到真实世界数字孪生的跨越。



## 🚀 核心特性
- **学术级算法**：采用 Masked-GAT 架构，将交通规则作为硬约束注入消息传递过程。
- **工业级实况**：接入高德 REST API，实现 WGS-84 与 GCJ-02 坐标系的高精度实时对齐。
- **数字孪生交互**：基于 Folium 实现交互式 3D 渲染，支持 AntPath 动态光流展示。
- **自适应泛化**：具备 Zero-shot 环境适应能力，可瞬间响应突发交通拥堵。

## 🛠️ 技术栈
- **Deep Learning**: PyTorch, PyTorch Geometric
- **Geospatial Analysis**: OSMnx, NetworkX, Folium
- **Interface**: Gradio
- **Data Source**: OpenStreetMap, Amap Traffic API

## 📈 视觉进化史
本项目经历了从基础张量化到 3D 数字孪生的四个阶段：
1. **v1.0**: 基础图拓扑验证
2. **v1.5**: 空间注意力热力分布
3. **v2.0**: 实况数据驱动推理
4. **v3.0**: 工业级交互终端

## 📥 快速开始
1. 克隆仓库：`git clone https://github.com/你的用户名/MIG-Planner.git`
2. 安装依赖：`pip install -r requirements.txt`
3. 运行演示：`python scripts/demo_v3.py`