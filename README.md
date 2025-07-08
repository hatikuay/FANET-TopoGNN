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

```bash
pip install -r requirements.txt

