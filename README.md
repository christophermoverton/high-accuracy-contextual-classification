# **Comparative Analysis of Transformer Architectures for Contextual Advertising Text Classification**

## **Overview**

This project evaluates multiple transformer-based NLP models to classify contextual advertising text, with a focus on health and wellness–related content. Building on prior exploratory research by Chris Vargo, the goal is to compare modern transformer architectures using a unified workflow and determine which approaches deliver the highest accuracy, stability, and practical value for real-world classification tasks.

Using **TensorFlow**, **ktrain**, and GPU-accelerated training, the project implements end-to-end pipelines for data preprocessing, model training, evaluation, and prediction. The result is a reproducible framework for benchmarking transformer models and identifying trade-offs between performance and computational efficiency.

---

## **Key Objectives**

* Evaluate and compare multiple transformer architectures for text classification
* Use ktrain’s streamlined API for training, fine-tuning, and prediction
* Explore preprocessing, tokenization, and learning rate strategies
* Quantitatively rank models using accuracy, loss curves, and validation metrics
* Produce actionable insights for selecting transformer models in contextual advertising applications

---

## **Data**

The dataset consists of contextual advertising text, including health and wellness–related messaging, adapted from prior research materials. Text samples include labels mapping to specific ad categories.

Data processing includes:

* Cleaning and normalization
* Tokenization via model-specific vocabularies
* Train/validation/test splitting

---

## **Methods**

### **Models Evaluated**

The project compares several transformer architectures (e.g., BERT variants, DistilBERT, RoBERTa, etc.) using identical training pipelines to ensure fair benchmarking.

### **Pipeline Tools**

* **TensorFlow 2.x** — model training backend
* **ktrain** — high-level wrapper for simplified fine-tuning
* **GPU acceleration** (Colab) — required for efficient training

### **Evaluation Metrics**

* Accuracy
* Validation loss
* Training convergence behavior
* Confusion matrices
* Error inspection on misclassified samples

---

## **Project Structure**

```
project/
│── data/                     # Input text data (not included in repo)
│── notebooks/
│     └── Ktrain_Final_Project.ipynb
│── models/                   # Saved model files (gitignored)
│── README.md
└── utils/                    # Preprocessing or helper functions
```

---

## **How to Run**

### **1. Set Colab Runtime**

```
Runtime > Change Runtime Type > GPU
```

### **2. Install Dependencies**

```bash
pip install ktrain tensorflow
```

### **3. Mount Google Drive (if using Colab)**

```python
from google.colab import drive
drive.mount('/content/drive')
```

### **4. Run the Notebook**

Open and execute all cells in:
`Ktrain_Final_Project (2).ipynb`

---

## **Results**

The project reports:

* Best-performing transformer model
* Impact of different fine-tuning strategies
* Comparative training times
* Error profiles and misclassification patterns

These findings illustrate how architecture choice influences real-world ad-text classification tasks, particularly where nuanced context is required.

---

## **Conclusion**

This project demonstrates the effectiveness of transformer architectures for contextual advertising classification and provides a reproducible framework for comparing models. The results offer guidance for selecting suitable architectures depending on accuracy needs, latency constraints, and available compute.

---

## **Future Work**

* Expand dataset variety for broader generalization
* Experiment with instruction-tuned models
* Incorporate explainability (e.g., SHAP, attention visualization)
* Evaluate large language model embeddings (e.g., InstructorXL, MPNet)


