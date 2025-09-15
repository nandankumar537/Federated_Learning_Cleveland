# Federated Learning with Flower and the Cleveland Dataset

This project demonstrates a Federated Learning (FL) setup using the Flower framework with the Cleveland Heart Disease dataset.  
The goal is to simulate multiple clients training collaboratively on subsets of the dataset without sharing raw data,  
preserving privacy while building a global model.

## Features
- Federated Learning setup using Flower (flwr) framework  
- Distributed training on the Cleveland Heart Disease dataset  
- Support for local client-server simulation  
- Configurable number of clients and training rounds  
- Training metrics such as accuracy, loss, precision, recall, and F1-score are logged  

## Project Structure
├── client.py    # Client training code
├── server.py    # Server aggregation and strategy code
├── cleveland.csv    # Cleveland dataset (CSV format or preprocessed)
├── requirements.txt    # Python dependencies
├── README.md    # Project documentation
└── converter.py    # Helper scripts for preprocessing, metrics, etc.
