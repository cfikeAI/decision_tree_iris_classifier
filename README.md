# Iris Decision Tree Classifier

A simple **Decision Tree Classifier** built using the classic [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris).  
This project demonstrates end-to-end supervised learning — from data loading and visualization to model training, evaluation, and prediction.

---

## Overview

The **Iris dataset** contains 150 samples of iris flowers across three species:
- *Iris Setosa*
- *Iris Versicolor*
- *Iris Virginica*

Each sample includes four numerical features:
- Sepal Length  
- Sepal Width  
- Petal Length  
- Petal Width  

The goal is to train a **Decision Tree** model to classify a flower into its correct species based on these features.

---

## Model Details

- **Algorithm:** Decision Tree Classifier (CART)
- **Library:** `scikit-learn`
- **Split:** 80% training / 20% testing
- **Metrics:** Accuracy, Confusion Matrix, Classification Report

---


# Clone repo
git clone https://github.com/cfikeAI/decision_tree_iris_classifier.git
cd iris-decision-tree

# Create venv
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
