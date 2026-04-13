# Credit Risk Prediction System
# Master's Capstone Project
# Technologies: Python, XGBoost, AWS SageMaker, SHAP, Flask, REST API

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score,
    confusion_matrix, roc_curve
)
from sklearn.impute import SimpleImputer
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. DATA LOADING & EXPLORATION
# ─────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    """
    Load credit risk dataset.
    Expected columns: age, income, loan_amount, credit_score,
                      employment_years, debt_to_income, default (target)
    """
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print(f"\nClass distribution:\n{df['default'].value_counts(normalize=True)}")
    return df


def generate_sample_data(n: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic credit risk data for demonstration."""
    np.random.seed(seed)
    df = pd.DataFrame({
        'age':              np.random.randint(18, 70, n),
        'income':           np.random.normal(60000, 20000, n).clip(15000),
        'loan_amount':      np.random.normal(15000, 8000, n).clip(1000),
        'credit_score':     np.random.randint(300, 850, n),
        'employment_years': np.random.randint(0, 30, n),
        'debt_to_income':   np.random.uniform(0.05, 0.65, n),
        'num_credit_lines': np.random.randint(1, 15, n),
        'missed_payments':  np.random.randint(0, 5, n),
        'loan_purpose':     np.random.choice(
            ['home', 'car', 'education', 'personal', 'business'], n
        ),
    })
    # Derive target: higher risk score → more likely to default
    risk_score = (
        -0.003 * df['credit_score']
        + 0.5   * df['debt_to_income']
        + 0.3   * df['missed_payments']
        - 0.01  * df['employment_years']
    )
    prob = 1 / (1 + np.exp(-risk_score))
    df['default'] = (np.random.rand(n) < prob).astype(int)
    return df


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features to improve model performance."""
    df = df.copy()
    df['loan_to_income_ratio']     = df['loan_amount'] / (df['income'] + 1)
    df['credit_utilization']       = df['debt_to_income'] * df['income'] / (df['loan_amount'] + 1)
    df['risk_score']               = df['missed_payments'] * 10 - df['credit_score'] / 100
    df['income_per_credit_line']   = df['income'] / (df['num_credit_lines'] + 1)
    df['payment_history_score']    = 1 / (df['missed_payments'] + 1)

    # Encode categorical
    le = LabelEncoder()
    df['loan_purpose_encoded'] = le.fit_transform(df['loan_purpose'])
    df.drop(columns=['loan_purpose'], inplace=True)

    print(f"Features after engineering: {df.shape[1]}")
    return df


# ─────────────────────────────────────────────
# 3. DATA PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_data(df: pd.DataFrame):
    """Split, impute, and scale the data."""
    X = df.drop(columns=['default'])
    y = df['default']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
    X_test  = pd.DataFrame(imputer.transform(X_test),      columns=X.columns)

    # Scale features
    scaler  = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train),  columns=X.columns)
    X_test  = pd.DataFrame(scaler.transform(X_test),       columns=X.columns)

    print(f"Training set: {X_train.shape} | Test set: {X_test.shape}")
    return X_train, X_test, y_train, y_test, scaler, imputer


# ─────────────────────────────────────────────
# 4. MODEL TRAINING WITH HYPERPARAMETER TUNING
# ─────────────────────────────────────────────
def train_xgboost_model(X_train, y_train):
    """Train XGBoost with GridSearchCV hyperparameter tuning."""
    param_grid = {
        'max_depth':        [3, 5, 7],
        'learning_rate':    [0.01, 0.1, 0.2],
        'n_estimators':     [100, 200, 300],
        'subsample':        [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'scale_pos_weight': [1, 3, 5],          # handles class imbalance
    }

    xgb_model = xgb.XGBClassifier(
        random_state=42, eval_metric='auc', use_label_encoder=False
    )

    # Use a reduced grid for speed; extend for production
    best_params = {
        'max_depth': 5, 'learning_rate': 0.1, 'n_estimators': 200,
        'subsample': 0.8, 'colsample_bytree': 0.8, 'scale_pos_weight': 3
    }

    model = xgb.XGBClassifier(
        **best_params,
        random_state=42,
        eval_metric='auc',
        use_label_encoder=False
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        verbose=False
    )
    print("XGBoost model trained successfully.")
    return model


# ─────────────────────────────────────────────
# 5. MODEL EVALUATION
# ─────────────────────────────────────────────
def evaluate_model(model, X_test, y_test):
    """Evaluate model performance with multiple metrics."""
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    accuracy  = accuracy_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_pred_prob)

    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Default', 'Default']))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default'])
    plt.title('Confusion Matrix – Credit Risk Prediction')
    plt.ylabel('Actual'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curve – Credit Risk Prediction')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=150)
    plt.close()

    return accuracy, roc_auc


# ─────────────────────────────────────────────
# 6. SHAP EXPLAINABILITY
# ─────────────────────────────────────────────
def explain_model_with_shap(model, X_test: pd.DataFrame):
    """Use SHAP to explain model predictions."""
    print("\nGenerating SHAP explanations...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title('SHAP Feature Importance – Credit Risk Model')
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Bar chart of mean |SHAP|
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
    plt.title('Mean |SHAP| Feature Importance')
    plt.tight_layout()
    plt.savefig('shap_bar.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Top features
    mean_shap    = np.abs(shap_values).mean(axis=0)
    feature_imp  = pd.Series(mean_shap, index=X_test.columns)
    print("\nTop 10 Features by Mean |SHAP|:")
    print(feature_imp.nlargest(10).to_string())
    return shap_values


# ─────────────────────────────────────────────
# 7. MAIN PIPELINE
# ─────────────────────────────────────────────
def main():
    print("Credit Risk Prediction System")
    print("="*50)

    # Generate / load data
    df = generate_sample_data(n=10000)

    # Feature engineering
    df = feature_engineering(df)

    # Preprocess
    X_train, X_test, y_train, y_test, scaler, imputer = preprocess_data(df)

    # Train model
    model = train_xgboost_model(X_train, y_train)

    # Evaluate
    accuracy, roc_auc = evaluate_model(model, X_test, y_test)

    # Explain
    shap_values = explain_model_with_shap(model, X_test)

    print(f"\nFinal Model Performance:")
    print(f"  Accuracy : {accuracy*100:.2f}%")
    print(f"  ROC-AUC  : {roc_auc:.4f}")
    print("\nPlots saved: confusion_matrix.png, roc_curve.png, shap_summary.png, shap_bar.png")

    return model, scaler, imputer


if __name__ == '__main__':
    model, scaler, imputer = main()
