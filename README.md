# FANET-TopoGNN

This repository contains the code and resources for **FANET-TopoGNN**, an academic research project developed by **Dr. Ercan Erkalkan**. The study focuses on the topological analysis of Flying Ad Hoc Networks (FANETs) using persistent homology and graph neural networks (GNNs).

## 🎯 Objective

The aim of this project is to analyze dynamic UAV networks in FANET environments through topological data analysis (TDA). By computing persistence images from UAV position data, the topological evolution of the network is captured and learned via a GNN model.

## 📁 Repository Structure

FANET-TopoGNN/
├── fanet_simulator.py # Data simulation and persistence image extraction
├── main.ipynb # Model training, evaluation, and visualization
├── best_model.pth # Pretrained PyTorch model
├── requirements.txt # Required Python packages
├── README.md # Project documentation (this file)
└── .gitattributes # Git LFS settings (optional)

## 📦 Dependencies

To install required dependencies:

pip install -r requirements.txt

Sample requirements.txt:

numpy
scipy
networkx
gudhi
persim
h5py
tqdm
matplotlib
torch

## ⚙️ Usage
1. Dataset Generation
Run the simulator to generate synthetic UAV topology data:

python fanet_simulator.py

his will produce a dataset file named fanet_topo_dataset.h5, which contains positional and topological snapshot data.

2. Model Training and Evaluation
Open and run main.ipynb in Jupyter Notebook or another compatible environment. It includes:

Loading the .h5 dataset

Training a GNN on persistence images

Evaluation metrics and result visualization

📚 Academic Context
This repository was prepared exclusively for academic purposes by:

Dr. Ercan Erkalkan
Department of Computer Engineering
Marmara University, Istanbul, Turkey
📧 ercan.erkalkan@marmara.edu.tr

This work is part of ongoing research in wireless UAV networks, persistent homology, and geometric deep learning.

If you use this code or dataset in your research, please cite appropriately or include an acknowledgment.