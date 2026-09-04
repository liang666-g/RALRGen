# RALRGen

## Overview

RALRGen is a retrieval-augmented framework for automatic app review response generation.

This repository accompanies the paper:

**Automating App Review Response Generation with Evolution-Aware Retrieval Augmentation**

## Requirements

Recommended environment:

- Python >= 3.10
- PyTorch >= 2.0
- CUDA >= 11.8 (optional, for GPU acceleration)
- sentence-transformers
- transformers
- faiss-cpu or faiss-gpu
- numpy
- pandas
- tqdm
- nltk
- rouge-score
- requests

We recommend using Conda to create an isolated environment:

```bash
conda create -n ralrgen python=3.10
conda activate ralrgen# RALRGen
