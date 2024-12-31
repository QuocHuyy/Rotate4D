# Knowledge Graph Embedding with the Special Orthogonal Group in Quaternion Space for Link Prediction

This repository contains the source code for **Rotate4D**, a knowledge graph embedding model that represents relations as rotations in four-dimensional space. The model was introduced in our recent publication.

---

## Overview
Rotate4D leverages the properties of the special orthogonal group in quaternion space to effectively model complex relational patterns in knowledge graphs. This repository provides the implementation of Rotate4D and the necessary scripts to reproduce the experimental results presented in the paper.

---

## Features
- **Four-Dimensional Rotations:** Encode relational patterns in 4D space for enhanced expressiveness.
- **Link Prediction Tasks:** Achieve state-of-the-art performance on benchmark datasets.
- **Highly Configurable:** Flexible hyperparameter settings for experimentation.

---

## Getting Started

### Prerequisites
Ensure you have the following dependencies installed:
- Python 3.8 or higher
- PyTorch 1.8 or higher
- Other dependencies listed in `requirements.txt`

Install the required packages using:
```bash
pip install -r requirements.txt
```

---

## Running Experiments

#### WordNet 18 (WN18)
```bash
bash runs.sh train Rotate4D wn18 0 0 512 256 1000 12.0 1.0 0.0001 80000 8 0 2 --disable_adv
```

#### Freebase 15k (FB15k)
```bash
bash runs.sh train Rotate4D FB15k 0 0 1024 256 1000 24.0 0.5 0.00005 150000 8 0 2
```

#### WordNet 18RR (WN18RR)
```bash
bash runs.sh train Rotate4D wn18rr 0 0 512 256 500 6.0 1.0 0.00005 80000 8 0.1 1 --disable_adv
```

#### Freebase 15k-237 (FB15k-237)
```bash
bash runs.sh train Rotate4D FB15k-237 0 0 1024 256 1000 12.0 1.0 0.00005 100000 8 0 2
```

---

## Citation
If you use Rotate4D in your research, please cite our paper:
```
@article{your_reference,
  title={Rotate4D: Knowledge graph embedding with the special orthogonal group in quaternion space for link prediction},
  author={Your Name and Co-authors},
  journal={Journal Name},
  year={2023},
  volume={XX},
  pages={XX-XX},
  doi={DOI}
}
```

---
