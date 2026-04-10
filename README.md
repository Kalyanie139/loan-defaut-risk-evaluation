# CreditIQ - Loan Default Risk Evaluation

A Machine Learning-powered Loan Risk Intelligence Platform that predicts the probability of loan default using financial and behavioral data, with explainability and document-based insights.

---

## Features

* Loan Default Prediction using ML model (LightGBM pipeline)
* Explainable AI (SHAP) for model interpretability
* OCR-based Document Analysis using Tesseract
* Interactive visualizations with Plotly
* Real-time risk scoring via Streamlit UI
* Secure API key handling using environment variables

---

## Tech Stack

* Frontend: Streamlit
* Backend: Python
* ML Model: LightGBM
* Explainability: SHAP
* OCR: Tesseract
* Visualization: Matplotlib, Plotly

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
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
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

### 5. Run the app

```
streamlit run app.py
```

---

## Model Details

* Algorithm: LightGBM
* Evaluation Metrics:

  * AUC-ROC: ~0.90
  * F1 Score: ~0.73
* Features: 40+ financial indicators

---

## Security

* API keys stored securely using `.env`
* `.gitignore` prevents sensitive data exposure

---

---

## Use Cases

* Banking and Financial Institutions
* Loan Risk Assessment
* Credit Scoring Systems
* FinTech Applications

---

## Future Improvements

* RBI guideline-based risk rules
* Model retraining pipeline
* Cloud deployment (AWS / GCP)
* User authentication system

---

## Author

Kalyani Somvanshi
GitHub: https://github.com/Kalyanie139

---

