# CreditIQ - Loan Default Risk Evaluation

CreditIQ is a Machine Learning-based risk intelligence platform designed to predict loan default probability using structured financial data and document-derived insights. The system combines predictive modeling, explainable AI, and document processing to assist in data-driven credit decisions.

---

## Problem Statement

Traditional credit evaluation systems rely heavily on limited financial indicators and manual verification. This often leads to:

* Inaccurate risk assessment
* Lack of interpretability
* Inefficient document processing
* Delayed decision-making

This project addresses these issues by building an automated, explainable, and scalable credit risk assessment system.

---

## Machine Learning Approach

### Data Processing

* Feature engineering on financial and behavioral attributes
* Handling missing values and categorical encoding
* Feature scaling using standardization techniques

### Model Selection

* Gradient Boosting framework using LightGBM
* Chosen for its performance on tabular data and ability to handle feature interactions

### Training Details

* Dataset size: ~8,000 loan records
* Features: 40+ engineered variables
* Target: Loan default classification

### Evaluation Metrics

* AUC-ROC: ~0.90
* AUC-PR: ~0.81
* F1 Score: ~0.73

These metrics indicate strong discrimination capability and balanced precision-recall performance.

---

## Explainability (Model Interpretability)

To address the "black-box" nature of ML models:

* SHAP (SHapley Additive exPlanations) is used
* Provides feature-level contribution for each prediction
* Enables transparency in decision-making

This is critical in financial systems where interpretability is required.

---

## System Architecture

1. User inputs financial details or uploads documents
2. OCR extracts relevant text from documents
3. Data is processed and transformed into model features
4. ML model predicts default probability
5. SHAP generates explanation for the prediction
6. Results are displayed via Streamlit dashboard

---

## OCR Integration

* Uses Tesseract for document processing
* Extracts structured data from uploaded financial documents
* Reduces manual data entry and improves automation

---

## Tech Stack

* Frontend: Streamlit
* Backend: Python
* Machine Learning: LightGBM, Scikit-learn
* Explainability: SHAP
* OCR: Tesseract
* Visualization: Plotly, Matplotlib

---

## Project Structure

```
loan-risk-app/
│── app.py
│── loan_pipeline.pkl
│── scaler.pkl
│── feature_cols.json
│── threshold.txt
│── requirements.txt
│── .env (ignored)
│── .gitignore
│── assets/
```

---

## Installation & Setup

### 1. Clone the repository

```
git clone https://github.com/Kalyanie139/loan-defaut-risk-evaluation.git
cd loan-risk-app
```

### 2. Create virtual environment

```
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Add environment variables

Create a `.env` file:

```
API_KEY=your_api_key_here
```

### 5. Run the application

```
streamlit run app.py
```

---

## Key Features

* Real-time loan default prediction
* Explainable AI with feature attribution
* Document-based data extraction
* Interactive visualization dashboard
* Modular and scalable pipeline

---

## Use Cases

* Banking and Financial Institutions
* Credit Risk Assessment Systems
* FinTech Applications
* Automated Loan Approval Systems

---

## Future Improvements

* Integration of RBI guideline-based rule engine
* Model retraining and monitoring pipeline
* Deployment using cloud infrastructure
* Enhanced feature engineering using external credit data

---

## Author

Kalyani Somvanshi
GitHub: https://github.com/Kalyanie139

---

## License

This project is for educational and demonstration purposes.
