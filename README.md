# 面向中文音乐的多模态多标签情绪与主题分类

本项目为论文《面向中文音乐的多模态多标签情绪与主题分类研究》的官方开源代码实现。

## 🌟 核心特点
- **多模态特征**：音频端采用 **MERT-v1-95M** 提取深度声学嵌入，歌词端采用 **BERT-base-chinese** 提取语义向量。
- **模型架构**：引入**跨模态注意力机制（Cross-Modal Attention）**进行全局语义交互，并设计**任务解耦架构（Task Decoupling）**，有效缓解多任务学习中的特征竞争问题。
- **严谨的实证分析**：包含多数类基线（Majority Baseline）、参数对齐消融实验、模态缺失分析以及跨数据集（PMEmo）迁移实验。

## 📁 核心代码结构
- `wenbenDanMoTai.py` / `yinpin.py`: 文本与音频单模态基线模型
- `mlpZaoQiRongHe.py`: 早期融合基线模型（Early Fusion）
- `c2LiangJiaGou.py`: 跨模态注意力与任务解耦架构（主模型）
- `bujiu.py`: 参数对齐的消融实验
- `LuBang.py`: 模态缺失鲁棒性分析
- `pmo1.py` ~ `pmo3.py`: PMEmo 公开数据集线性探测与迁移实验
- `TongJiShiYan.py`: 统计显著性检验（Wilcoxon 检验等）

## 🛠️ 环境依赖
主要依赖 PyTorch 与 HuggingFace 生态：
```bash
pip install torch torchaudio transformers librosa scikit-learn pandas numpy scipy
