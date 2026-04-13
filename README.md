# 🏦 Credit Risk Prediction System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange.svg)](https://xgboost.readthedocs.io/)
[![AWS](https://img.shields.io/badge/AWS-SageMaker-yellow.svg)](https://aws.amazon.com/sagemaker/)
[![Flask](https://img.shields.io/badge/Flask-API-green.svg)](https://flask.palletsprojects.com/)

> **Master's Capstone Project** | University of New Haven | Data Science

An end-to-end machine learning system for predicting credit default risk with 92% accuracy. Built with XGBoost, deployed on AWS SageMaker, and served via Flask REST API with SHAP-based explainability.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Technologies Used](#-technologies-used)
- [Model Performance](#-model-performance)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Endpoints](#-api-endpoints)
- [Project Structure](#-project-structure)
- [Results](#-results)

---

## 🎯 Project Overview

### Problem Statement
Financial institutions face significant risk when lending money. Accurately predicting which borrowers are likely to default on loans is crucial for:
- **Risk Management**: Minimize financial losses
- **Fair Lending**: Make data-driven, unbiased decisions
- **Regulatory Compliance**: Meet banking regulations and audit requirements

### Solution
This project develops a production-ready credit risk assessment system that:
1. **Predicts default probability** for loan applicants with 92% accuracy
2. **Explains predictions** using SHAP values for regulatory compliance
3. **Scales to production** with AWS SageMaker deployment
4. **Provides real-time API** for integration with banking systems

### Business Impact
- **22% improvement** in risk decisioning accuracy
- **Reduced loan defaults** through better applicant screening
- **Faster decisions**: Real-time API responses (<100ms)
- **Explainable AI**: SHAP values satisfy regulatory requirements

---

## ✨ Key Features

### 1. Advanced Feature Engineering
- **Loan-to-income ratio** calculation
- **Credit utilization** metrics
- **Payment history scoring**
- **Risk score derivation** from multiple factors
- **Categorical encoding** for loan purpose

### 2. XGBoost Model with Hyperparameter Tuning
- **GridSearchCV** for optimal parameters
- **Class imbalance handling** with scale_pos_weight
- **Cross-validation** for robust evaluation
- **Regularization** to prevent overfitting

### 3. Model Explainability (SHAP)
- **Feature importance** visualization
- **Individual prediction** explanations
- **Regulatory compliance** documentation
- **Stakeholder communication** tools

### 4. Production Deployment
- **AWS SageMaker** training and hosting
- **Flask REST API** for real-time predictions
- **Docker containerization** for portability
- **Batch prediction** support
- **Model monitoring** and drift detection

---

## 🏗️ Architecture

```
┌─────────────────┐
│  Data Sources   │
│ (Applicant Data)│
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Feature Engineering Pipeline   │
│  • Data cleaning & imputation   │
│  • Feature creation             │
│  • Scaling & encoding           │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   XGBoost Training (SageMaker)  │
│  • Hyperparameter tuning        │
│  • Cross-validation             │
│  • Model serialization          │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  SHAP Explainability Analysis   │
│  • Feature importance           │
│  • Individual explanations      │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│     Flask REST API Serving      │
│  • /predict endpoint            │
│  • /batch_predict endpoint      │
│  • /health check                │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Banking Application / Users   │
└─────────────────────────────────┘
```

---

## 🛠️ Technologies Used

| Category | Technologies |
|----------|-------------|
| **ML Framework** | XGBoost, Scikit-learn |
| **Cloud Platform** | AWS SageMaker, S3, EC2 |
| **API Framework** | Flask, REST |
| **Explainability** | SHAP |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Deployment** | Docker, Kubernetes |
| **Model Monitoring** | MLflow |

---

## 📊 Model Performance

### Classification Metrics
- **Accuracy**: 92%
- **ROC-AUC**: 0.94
- **Precision**: 89%
- **Recall**: 91%
- **F1-Score**: 90%

### Top Risk Factors (SHAP Analysis)
1. **Credit Score** (-0.45)
2. **Debt-to-Income Ratio** (+0.38)
3. **Missed Payments** (+0.32)
4. **Employment Years** (-0.21)
5. **Loan Amount** (+0.18)

---

## 💻 Installation

### Prerequisites
- Python 3.8+
- AWS Account (for SageMaker deployment)
- Docker (optional, for containerization)

### Setup

```bash
# Clone repository
git clone https://github.com/hemanthsaig23/credit-risk-prediction.git
cd credit-risk-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Train Model

```bash
python credit_risk_model.py
```

**Output:**
- Trained XGBoost model
- Performance metrics
- SHAP visualizations
- Saved model artifacts

### 2. Start Flask API

```bash
python app.py
```

API will be available at `http://localhost:5000`

### 3. Make Predictions

```python
import requests

# Single prediction
data = {
    "age": 35,
    "income": 75000,
    "loan_amount": 20000,
    "credit_score": 680,
    "employment_years": 8,
    "debt_to_income": 0.35,
    "num_credit_lines": 5,
    "missed_payments": 1,
    "loan_purpose": "home"
}

response = requests.post('http://localhost:5000/predict', json=data)
print(response.json())
```

**Response:**
```json
{
    "prediction": 0,
    "default_probability": 0.1234,
    "risk_level": "LOW",
    "risk_label": "No Default",
    "top_risk_factors": {
        "credit_score": 0.42,
        "debt_to_income": 0.31,
        "missed_payments": 0.28
    }
}
```

---

## 📡 API Endpoints

### `GET /health`
Health check endpoint

**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "timestamp": "2026-04-13T13:00:00Z"
}
```

### `POST /predict`
Single applicant prediction

**Request Body:**
```json
{
    "age": 35,
    "income": 75000,
    "loan_amount": 20000,
    "credit_score": 680,
    "employment_years": 8,
    "debt_to_income": 0.35,
    "num_credit_lines": 5,
    "missed_payments": 1
}
```

### `POST /batch_predict`
Batch predictions for multiple applicants

**Request Body:**
```json
[
    {"age": 35, "income": 75000, ...},
    {"age": 42, "income": 60000, ...}
]
```

---

## 📁 Project Structure

```
credit-risk-prediction/
├── credit_risk_model.py     # Main training script
├── app.py                   # Flask REST API
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── models/                 # Saved model artifacts
├── data/                   # Data files
├── plots/                  # Generated visualizations
└── tests/                  # Unit tests
```

---

## 📈 Results

### Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

### ROC Curve
![ROC Curve](roc_curve.png)

### SHAP Feature Importance
![SHAP Summary](shap_summary.png)

---

## 🎓 Academic Context

**Project Type:** Master's Capstone Project  
**Institution:** University of New Haven  
**Program:** M.S. in Data Science  
**Duration:** Aug 2023 - May 2025  

### Learning Outcomes
- Production ML system development
- Cloud deployment (AWS SageMaker)
- Model explainability for regulated industries
- REST API design and implementation
- End-to-end MLOps practices

---

## 👤 Author

**Hemanth Sai Gogineni**  
📧 hemanthsaigogineni@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/hemanthsaig23) | [GitHub](https://github.com/hemanthsaig23)

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- University of New Haven - Data Science Program
- AWS for SageMaker credits
- XGBoost and SHAP open-source communities
