# Flask REST API for Credit Risk Prediction
# Technologies: Flask, XGBoost, SHAP, AWS SageMaker

from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import joblib
import logging
from datetime import datetime

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load trained model and preprocessors (saved during training)
try:
    model   = joblib.load('credit_risk_model.pkl')
    scaler  = joblib.load('scaler.pkl')
    imputer = joblib.load('imputer.pkl')
    explainer = shap.TreeExplainer(model)
    logger.info("Model and preprocessors loaded successfully.")
except FileNotFoundError:
    logger.warning("Model files not found. Run credit_risk_model.py first.")
    model = scaler = imputer = explainer = None


FEATURES = [
    'age', 'income', 'loan_amount', 'credit_score', 'employment_years',
    'debt_to_income', 'num_credit_lines', 'missed_payments',
    'loan_purpose_encoded', 'loan_to_income_ratio', 'credit_utilization',
    'risk_score', 'income_per_credit_line', 'payment_history_score'
]


def engineer_features(data: dict) -> pd.DataFrame:
    """Apply same feature engineering as during training."""
    df = pd.DataFrame([data])
    df['loan_to_income_ratio']   = df['loan_amount'] / (df['income'] + 1)
    df['credit_utilization']     = df['debt_to_income'] * df['income'] / (df['loan_amount'] + 1)
    df['risk_score']             = df['missed_payments'] * 10 - df['credit_score'] / 100
    df['income_per_credit_line'] = df['income'] / (df['num_credit_lines'] + 1)
    df['payment_history_score']  = 1 / (df['missed_payments'] + 1)
    purpose_map = {'home': 0, 'car': 1, 'education': 2, 'personal': 3, 'business': 4}
    df['loan_purpose_encoded'] = df.get('loan_purpose', pd.Series(['personal'])).map(purpose_map).fillna(3)
    return df[FEATURES]


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.utcnow().isoformat()
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict credit default risk.
    Expected JSON body:
    {
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
    """
    if model is None:
        return jsonify({'error': 'Model not loaded. Run training first.'}), 503

    try:
        data = request.get_json(force=True)
        required = ['age', 'income', 'loan_amount', 'credit_score',
                    'employment_years', 'debt_to_income', 'num_credit_lines', 'missed_payments']
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({'error': f'Missing fields: {missing}'}), 400

        # Feature engineering
        X = engineer_features(data)

        # Impute & scale
        X_imp    = pd.DataFrame(imputer.transform(X), columns=FEATURES)
        X_scaled = pd.DataFrame(scaler.transform(X_imp), columns=FEATURES)

        # Predict
        prob        = float(model.predict_proba(X_scaled)[0][1])
        prediction  = int(model.predict(X_scaled)[0])
        risk_level  = 'HIGH' if prob > 0.7 else ('MEDIUM' if prob > 0.4 else 'LOW')

        # SHAP explanation
        shap_vals   = explainer.shap_values(X_scaled)
        top_features = pd.Series(
            np.abs(shap_vals[0]), index=FEATURES
        ).nlargest(5).to_dict()

        logger.info(f"Prediction: {prediction} | Prob: {prob:.4f} | Risk: {risk_level}")

        return jsonify({
            'prediction':       prediction,
            'default_probability': round(prob, 4),
            'risk_level':       risk_level,
            'risk_label':       'Default' if prediction == 1 else 'No Default',
            'top_risk_factors': top_features,
            'timestamp':        datetime.utcnow().isoformat()
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint.
    Expected: JSON array of applicant records.
    """
    if model is None:
        return jsonify({'error': 'Model not loaded.'}), 503
    try:
        records = request.get_json(force=True)
        results = []
        for rec in records:
            X        = engineer_features(rec)
            X_imp    = pd.DataFrame(imputer.transform(X),       columns=FEATURES)
            X_scaled = pd.DataFrame(scaler.transform(X_imp),   columns=FEATURES)
            prob     = float(model.predict_proba(X_scaled)[0][1])
            pred     = int(model.predict(X_scaled)[0])
            results.append({
                'default_probability': round(prob, 4),
                'prediction':         pred,
                'risk_level': 'HIGH' if prob > 0.7 else ('MEDIUM' if prob > 0.4 else 'LOW')
            })
        return jsonify({'predictions': results, 'count': len(results)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
