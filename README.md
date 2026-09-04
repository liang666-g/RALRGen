# RALRGen

## 项目简介

RALRGen（Retrieval-Augmented LLM-based Response Generation）是一个面向移动应用用户评论的自动回复生成框架。

该项目对应论文：

**Automating App Review Response Generation with Evolution-Aware Retrieval Augmentation**

## 运行环境

推荐使用以下环境：

- Python >= 3.10
- PyTorch >= 2.0
- CUDA >= 11.8（使用 GPU 时）
- sentence-transformers
- transformers
- faiss-cpu / faiss-gpu
- numpy
- pandas
- tqdm
- nltk
- rouge-score
- requests

推荐使用 Conda 创建独立环境：

```bash
conda create -n ralrgen python=3.10
conda activate ralrgen# RALRGen
