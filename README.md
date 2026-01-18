# Rotate4D

<div align="center">

**Knowledge Graph Embedding with the Special Orthogonal Group in Quaternion Space for Link Prediction**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-ee4c2c.svg)](https://pytorch.org/)

</div>

---

## 📋 Table of Contents

- [Introduction](#-introduction)
- [Model Architecture](#-model-architecture)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Usage](#-usage)
- [Experimental Results](#-experimental-results)
- [Citation](#-citation)
- [Contact](#-contact)

---

## 🎯 Introduction

**Rotate4D** is an advanced Knowledge Graph Embedding model that utilizes rotations in 4-dimensional quaternion space to model complex relational patterns. This model is specifically designed for link prediction tasks on knowledge graphs.

### Key Features

- 🔄 **4D Rotations**: Utilizes the special orthogonal group in quaternion space to represent relations
- 📊 **High Performance**: Achieves state-of-the-art results on benchmark datasets
- ⚙️ **Flexible**: Easy to customize hyperparameters for experimentation
- 🎓 **Strong Expressiveness**: Effectively models complex relational patterns such as symmetry, anti-symmetry, and transitivity

---

## 🏗 Model Architecture

Rotate4D extends the idea of rotation-based models like RotatE by:

1. **Entity and Relation Representation in 4D Space**: Each entity and relation is embedded into quaternion space
2. **Using Quaternion Rotations**: Relations are modeled as rotations in 4-dimensional space
3. **Scoring Function**: Measures the plausibility of triples (head, relation, tail) based on the distance after rotation

Basic formula:
```
h ∘ r ≈ t
```
where `∘` represents quaternion rotation.

---

## 🚀 Installation

### System Requirements

- Python 3.8 or higher
- PyTorch 1.8 or higher
- CUDA (recommended for faster training)

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/QuocHuyy/Rotate4D.git
cd Rotate4D
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
Rotate4D/
├── codes/              # Main source code
│   ├── model.py       # Rotate4D model definition
│   ├── dataloader.py  # Data processing
│   └── run.py         # Training and evaluation script
├── data/              # Experimental datasets
│   ├── wn18/
│   ├── FB15k/
│   ├── wn18rr/
│   └── FB15k-237/
├── runs.sh            # Experiment execution script
├── requirements.txt   # Dependencies
└── README.md         # This file
```

---

## 💻 Usage

### Basic Training

The `runs.sh` file provides commands to train the model on different datasets:

#### 1. WordNet 18 (WN18)
```bash
bash runs.sh train Rotate4D wn18 0 0 512 256 1000 12.0 1.0 0.0001 80000 8 0 2 --disable_adv
```

#### 2. Freebase 15k (FB15k)
```bash
bash runs.sh train Rotate4D FB15k 0 0 1024 256 1000 24.0 0.5 0.00005 150000 8 0 2
```

#### 3. WordNet 18RR (WN18RR)
```bash
bash runs.sh train Rotate4D wn18rr 0 0 512 256 500 6.0 1.0 0.00005 80000 8 0.1 1 --disable_adv
```

#### 4. Freebase 15k-237 (FB15k-237)
```bash
bash runs.sh train Rotate4D FB15k-237 0 0 1024 256 1000 12.0 1.0 0.00005 100000 8 0 2
```

### Parameter Explanation

Training command parameters in order:
1. `train`: Mode (train/test)
2. `Rotate4D`: Model name
3. `dataset`: Dataset name
4. `gpu`: GPU ID
5. `save_id`: Checkpoint ID
6. `hidden_dim`: Embedding dimension
7. `batch_size`: Batch size
8. `negative_sample_size`: Number of negative samples
9. `gamma`: Margin in loss function
10. `alpha`: Adversarial temperature
11. `learning_rate`: Learning rate
12. `max_steps`: Maximum training steps
13. `test_batch_size`: Test batch size
14. `regularization`: Regularization coefficient
15. `adversarial_temperature`: Temperature for adversarial sampling

### Custom Training

You can customize training by modifying parameters in `runs.sh` or running directly from Python:

```python
python codes/run.py --do_train \
    --model Rotate4D \
    --dataset wn18 \
    --hidden_dim 512 \
    --batch_size 256 \
    --learning_rate 0.0001 \
    --max_steps 80000
```

---

## 📊 Experimental Results

### Performance on Benchmark Datasets

| Dataset | MRR | Hits@1 | Hits@3 | Hits@10 |
|---------|-----|--------|--------|---------|
| WN18 | - | - | - | - |
| FB15k | - | - | - | - |
| WN18RR | - | - | - | - |
| FB15k-237 | - | - | - | - |

*Note: Update with your specific experimental results*

---

## 📝 Citation

If you use Rotate4D in your research, please cite our paper:

```bibtex
@article{rotate4d2024,
  title={Rotate4D: Knowledge graph embedding with the special orthogonal group in quaternion space for link prediction},
  author={Your Name and Co-authors},
  journal={Conference/Journal Name},
  year={2024},
  volume={XX},
  pages={XX-XX}
}
```

---

## 📄 License

This project is released under the [Apache License 2.0](LICENSE).

---

## 🤝 Contributing

We welcome all contributions! If you'd like to:

- 🐛 Report bugs
- 💡 Suggest new features
- 🔧 Submit pull requests

Please create an issue or pull request on GitHub.

---

## 📧 Contact

- **GitHub**: [@QuocHuyy](https://github.com/QuocHuyy)
- **Email**: your.email@example.com (update with your email)

---

## 🙏 Acknowledgments

Thanks to the Knowledge Graph Embedding research community and the authors of predecessor models like RotatE and QuatE for laying the foundation for this work.

---

<div align="center">

**⭐ If you find this project useful, please give it a star! ⭐**

Made with ❤️ by QuocHuyy

</div>
