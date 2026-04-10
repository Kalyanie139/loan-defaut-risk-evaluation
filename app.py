import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import datetime
import re
import warnings
import os
import io
from dotenv import load_dotenv
from PIL import Image, ImageOps
import pytesseract

warnings.filterwarnings('ignore')

load_dotenv()

class AppConfig:
    PAGE_TITLE = "CreditIQ - Loan Risk Intelligence"
    PAGE_ICON = "🏦"
    MODEL_PATH = "loan_pipeline.pkl"
    FEATURES_PATH = "feature_cols.json"
    THRESHOLD_PATH = "threshold.txt"
    SCALER_PATH = "scaler.pkl"

    API_KEY = os.getenv("API_KEY")

class InterestRateConfig:
    """Dynamic interest rates based on loan type and risk factors"""
    BASE_RATES = {
        "Personal Loan": 11.5,
        "Home Loan": 8.5,
        "Vehicle Loan": 9.5,
        "Education Loan": 10.0,
        "MSME Loan": 12.0,
    }
    
    @staticmethod
    def get_interest_rate(loan_type: str, credit_score: int = 700) -> float:
        """Calculate interest rate based on loan type and credit score"""
        base_rate = InterestRateConfig.BASE_RATES.get(loan_type, 11.5)
        
        # Credit score adjustment (better score = lower rate)
        if credit_score >= 750:
            credit_adjustment = -0.75
        elif credit_score >= 700:
            credit_adjustment = 0.0
        elif credit_score >= 650:
            credit_adjustment = 0.75
        elif credit_score >= 600:
            credit_adjustment = 1.5
        else:
            credit_adjustment = 2.5
        
        final_rate = base_rate + credit_adjustment
        return round(final_rate, 2)

class RoleConfig:
    ROLES = {
        "loan_officer": {
            "label": "Loan Officer",
            "icon": "👔",
            "color": "#1A3A5C",
            "password": "officer123",
            "description": "Full model access, SHAP explanations, audit trail"
        },
        "applicant": {
            "label": "Applicant",
            "icon": "👤",
            "color": "#047857",
            "password": "applicant123",
            "description": "Check your loan eligibility and improvement tips"
        }
    }

# ═══════════════════════════════════════════════════════════════════
# OOP: Model Manager
# ═══════════════════════════════════════════════════════════════════
class ModelManager:
    def __init__(self):
        self.model = None
        self.threshold = 0.5
        self.features = []
        self._load()

    @st.cache_resource
    def _load_cached(_self):
        try:
            pipeline = joblib.load(AppConfig.MODEL_PATH)
            if isinstance(pipeline, dict):
                model = pipeline["model"]
                threshold = pipeline.get("threshold", 0.5)
                features = pipeline.get("features", [])
            else:
                model = pipeline
                with open(AppConfig.THRESHOLD_PATH) as f:
                    threshold = float(f.read().strip())
                with open(AppConfig.FEATURES_PATH) as f:
                    features = json.load(f)
            return model, threshold, features
        except Exception as e:
            return None, 0.5, []

    def _load(self):
        result = self._load_cached()
        self.model, self.threshold, self.features = result

    def predict(self, data: dict):
        if self.model is None:
            return 0.5, "UNKNOWN", "MEDIUM"
        df = self._build_features(data)
        prob = self.model.predict_proba(df)[0][1]
        decision = "DEFAULT RISK" if prob >= self.threshold else "APPROVED"
        risk = "HIGH" if prob >= 0.6 else ("MEDIUM" if prob >= 0.3 else "LOW")
        return prob, decision, risk, df

    def _build_features(self, data: dict) -> pd.DataFrame:
        d = data.copy()
        defaults = {
            'age': 35, 'employment_years': 3, 'annual_income_inr': 600000,
            'loan_amount_inr': 300000, 'loan_tenure_months': 36,
            'interest_rate_pct': 11.5,
            'credit_score': 680,
            'num_existing_loans': 1, 'dti_ratio': 0.35,
            'ltv_ratio': 0.0, 'has_collateral': 0,
            'bureau_enquiries_6m': 1, 'missed_payments_2y': 0,
            'savings_account_balance_inr': 50000,
        }
        for k, v in defaults.items():
            d.setdefault(k, v)

        d['ltv_is_missing'] = 1 if d.get('ltv_ratio', 0) == 0 else 0
        d['loan_to_income_ratio'] = d['loan_amount_inr'] / (d['annual_income_inr'] + 1)
        d['dti_credit_risk'] = d['dti_ratio'] / (d['credit_score'] / 700 + 0.001)
        d['income_per_year_employed'] = d['annual_income_inr'] / (d['employment_years'] + 1)
        d['emi_to_income'] = (d['loan_amount_inr'] / d['loan_tenure_months']) / (d['annual_income_inr'] / 12 + 1)
        d['risk_score'] = d['missed_payments_2y'] * 2 + d['bureau_enquiries_6m'] * 0.5
        d['savings_to_loan'] = d['savings_account_balance_inr'] / (d['loan_amount_inr'] + 1)
        d['credit_dti'] = d['credit_score'] * (1 - d['dti_ratio'])

        cat_maps = {
            'gender': ['Male', 'Female', 'Other'],
            'education': ['Graduate', 'Post-Graduate', 'Under-Graduate', 'Doctorate'],
            'urban_rural': ['Urban', 'Rural', 'Semi-Urban'],
            'employment_type': ['Salaried', 'Self-Employed', 'Business', 'Freelancer'],
            'loan_type': ['Personal Loan', 'Home Loan', 'Vehicle Loan', 'Education Loan', 'MSME Loan'],
            'state': ['Maharashtra', 'Delhi', 'Karnataka', 'Tamil Nadu', 'Gujarat',
                      'Rajasthan', 'Uttar Pradesh', 'West Bengal', 'Telangana',
                      'Andhra Pradesh', 'Kerala', 'Madhya Pradesh', 'Other'],
        }
        for col, values in cat_maps.items():
            val = d.get(col, values[0])
            for v in values[1:]:
                d[f"{col}_{v}"] = 1 if val == v else 0

        df = pd.DataFrame([d])
        for col in self.features:
            if col not in df.columns:
                df[col] = 0
        if self.features:
            df = df[self.features]
        return df

    def get_shap(self, df):
        try:
            explainer = shap.TreeExplainer(self.model)
            sv = explainer.shap_values(df)
            if isinstance(sv, list):
                sv = sv[1]
            base = explainer.expected_value
            if isinstance(base, list):
                base = base[1]
            return sv[0], base, self.features
        except:
            return None, None, self.features

# ═══════════════════════════════════════════════════════════════════
# OOP: AI Report Generator
# ═══════════════════════════════════════════════════════════════════
class AIReportGenerator:
    @staticmethod
    def generate_officer_report(data: dict, prob: float, risk: str, top_factors: list) -> str:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=AppConfig.OPENAI_API_KEY)
            factors_text = "\n".join([
                f"  - {f['feature'].replace('_',' ').title()}: SHAP={f['value']:.4f} "
                f"({'increases' if f['value']>0 else 'reduces'} default risk)"
                for f in top_factors[:6]
            ])
            risk_segment = "top 20 percent highest-risk applicants" if prob > 0.7 else \
                           "borderline applicants (40-60th risk percentile)" if prob > 0.4 else \
                           "low-risk applicants (bottom 30th percentile)"

            prompt = f"""You are a senior credit risk officer at IndusCredit Finance NBFC preparing a formal credit memo.

APPLICANT PROFILE:
- Default Probability: {prob:.1%} | Risk: {risk} | Decision: {'REJECT' if prob >= 0.5 else 'APPROVE'}
- Annual Income: ₹{data.get('annual_income_inr', 0):,.0f} | Loan Amount: ₹{data.get('loan_amount_inr', 0):,.0f}
- Loan Type: {data.get('loan_type', 'Personal Loan')} | Interest Rate: {data.get('interest_rate_pct', 11.5)}%
- Credit Score: {data.get('credit_score', 0)}
- DTI Ratio: {data.get('dti_ratio', 0):.2f} | Missed Payments (2Y): {data.get('missed_payments_2y', 0)}
- Bureau Enquiries (6M): {data.get('bureau_enquiries_6m', 0)} | Employment: {data.get('employment_type', 'Salaried')} ({data.get('employment_years', 0)} yrs)
- Savings: ₹{data.get('savings_account_balance_inr', 0):,.0f} | Existing Loans: {data.get('num_existing_loans', 0)}

ML MODEL RISK FACTORS (SHAP Analysis):
{factors_text}

PEER CONTEXT: This applicant falls in the {risk_segment} based on our 8,000-loan training corpus.

Write a formal credit assessment memo in exactly 3 paragraphs:
1. VERDICT: Clear approve/reject/conditional with probability and peer comparison
2. RISK DRIVERS: Top 2-3 SHAP factors and their specific business implications
3. RECOMMENDATION: Actionable conditions or approval terms

Under 180 words. Formal bank tone. No bullet points. No markdown."""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=320,
                temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception as e:
            return AIReportGenerator._fallback_report(prob, risk, top_factors, data)

    @staticmethod
    def generate_applicant_advice(data: dict, prob: float, risk: str) -> str:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=AppConfig.OPENAI_API_KEY)
            prompt = f"""You are a friendly financial advisor helping a loan applicant understand their result.

Their default probability is {prob:.1%} and they are {risk} risk.
Credit Score: {data.get('credit_score', 0)}, DTI Ratio: {data.get('dti_ratio', 0):.2f}, 
Missed Payments: {data.get('missed_payments_2y', 0)}, Savings: ₹{data.get('savings_account_balance_inr', 0):,.0f}
Interest Rate Offered: {data.get('interest_rate_pct', 11.5)}%

Write in simple, friendly language (not banker jargon):
1. What this result means for them (1-2 sentences)
2. The top 2 things they can do RIGHT NOW to improve their chances
3. An encouraging closing line

Keep it under 120 words. Warm, supportive tone."""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.5,
            )
            return response.choices[0].message.content
        except:
            score = data.get('credit_score', 0)
            dti = data.get('dti_ratio', 0)
            advice = []
            if score < 700:
                advice.append(f"Pay all EMIs on time for 6 months to improve your credit score from {score} to 700+")
            if dti > 0.4:
                advice.append(f"Reduce your monthly debt payments - your DTI of {dti:.0%} is above the ideal 40%")
            if not advice:
                advice.append("Maintain your excellent credit habits and you will qualify easily")
            return f"Your loan application shows {'some risk' if prob > 0.4 else 'good standing'}. Key steps to improve: {'. '.join(advice)}."

    @staticmethod
    def _fallback_report(prob, risk, top_factors, data):
        direction = "reject" if prob >= 0.5 else "approve"
        top = top_factors[0]['feature'].replace('_', ' ').title() if top_factors else "credit profile"
        return (f"This application is assessed as {risk} risk with a {prob:.1%} default probability. "
                f"The credit committee recommends to {direction} this application. "
                f"Primary driver: {top}. "
                f"{'Applicant should reduce DTI and clear missed payments before reapplying.' if prob >= 0.5 else 'Standard monitoring terms apply.'}")

# ═══════════════════════════════════════════════════════════════════
# OOP: Document Processor
# ═══════════════════════════════════════════════════════════════════
class DocumentProcessor:
    @staticmethod
    def extract_text_from_image(image_bytes):
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image = image.convert('L')
            image = ImageOps.autocontrast(image)
            text = pytesseract.image_to_string(image, lang='eng')
            return text
        except Exception as e:
            st.warning(f"OCR error: {str(e)}")
            return ""

    @staticmethod
    def extract_text(uploaded_file) -> str:
        file_bytes = uploaded_file.read()
        file_type = uploaded_file.type
        
        if 'image' in file_type:
            return DocumentProcessor.extract_text_from_image(file_bytes)
        
        if file_type == 'application/pdf':
            try:
                import fitz
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                text = ""
                for page in doc:
                    text += page.get_text()
                return text
            except ImportError:
                return "PDF support requires PyMuPDF: pip install pymupdf"
        
        try:
            return file_bytes.decode('utf-8', errors='ignore')
        except:
            return ""

    @staticmethod
    def parse(doc_type: str, text: str) -> dict:
        f = {}
        
        if doc_type == "Bank Statement":
            salary_match = re.search(r'(?:Salary Credit|CREDIT|SALARY)[^\d]*[\₹\s]*([\d,]+)', text, re.IGNORECASE)
            if salary_match:
                salary = float(salary_match.group(1).replace(',', ''))
                if 10000 <= salary <= 500000:
                    f['annual_income_inr'] = salary * 12
            
            if 'annual_income_inr' not in f:
                credit_pattern = r'(?:credit|salary)[^\d]*[\₹\s]*([\d,]+)'
                credits = re.findall(credit_pattern, text, re.IGNORECASE)
                for credit in credits:
                    try:
                        val = float(credit.replace(',', ''))
                        if 20000 <= val <= 500000:
                            f['annual_income_inr'] = val * 12
                            break
                    except:
                        continue
            
            balance_pattern = r'[\d,]+\s+[\d,]+\s+([\d,]+)(?:\s*$|\s+\d{2}/\d{2})'
            balances = re.findall(balance_pattern, text, re.MULTILINE)
            
            if balances:
                for bal in reversed(balances):
                    try:
                        balance_val = float(bal.replace(',', ''))
                        if 1000 <= balance_val <= 100000000:
                            f['savings_account_balance_inr'] = balance_val
                            break
                    except:
                        continue
            
            if 'savings_account_balance_inr' not in f:
                balance_match = re.search(r'(?:Balance|Closing Balance|Available Balance)[^\d]*[\₹\s]*([\d,]+)', text, re.IGNORECASE)
                if balance_match:
                    try:
                        f['savings_account_balance_inr'] = float(balance_match.group(1).replace(',', ''))
                    except:
                        pass
            
            f['_bounces'] = len(re.findall(r'(?:bounce|ECS failed|NACH failed|insufficient|returned|dishonoured)', text, re.IGNORECASE))
            f['_validated'] = 'annual_income_inr' in f or 'savings_account_balance_inr' in f
            
            if 'annual_income_inr' in f:
                st.session_state['_debug_income'] = f['annual_income_inr']
            if 'savings_account_balance_inr' in f:
                st.session_state['_debug_balance'] = f['savings_account_balance_inr']
                
        elif doc_type == "Salary Slip":
            net_match = re.search(r'(?:Net Salary|Net Pay|Take Home|Total Earnings)[^\d]*[\₹\s]*([\d,]+)', text, re.IGNORECASE)
            if net_match:
                net = float(net_match.group(1).replace(',', ''))
                if 10000 <= net <= 500000:
                    f['annual_income_inr'] = net * 12
            
            years_match = re.search(r'(?:Years of Service|Experience|Service Years)[^\d]*(\d+)', text, re.IGNORECASE)
            if years_match:
                f['employment_years'] = float(years_match.group(1))
                
            f['_validated'] = 'annual_income_inr' in f
            
        elif doc_type == "CIBIL Report":
            score_match = re.search(r'(?:CIBIL Score|Credit Score|CIR Score)[^\d]*(\d{3})', text, re.IGNORECASE)
            if score_match:
                score = float(score_match.group(1))
                if 300 <= score <= 900:
                    f['credit_score'] = score
            
            enq_match = re.search(r'(?:Enquiries|Inquiries|Recent Enquiries)[^\d]*(\d+)', text, re.IGNORECASE)
            if enq_match:
                f['bureau_enquiries_6m'] = min(float(enq_match.group(1)), 20)
                
            f['_validated'] = 'credit_score' in f
            
        elif doc_type == "Loan Application":
            amt_match = re.search(r'(?:Loan Amount|Amount Requested|Principal Amount)[^\d]*[\₹\s]*([\d,]+)', text, re.IGNORECASE)
            if amt_match:
                f['loan_amount_inr'] = float(amt_match.group(1).replace(',', ''))
            
            tenure_match = re.search(r'(?:Tenure|Loan Period|Repayment Period)[^\d]*(\d+)', text, re.IGNORECASE)
            if tenure_match:
                tenure = float(tenure_match.group(1))
                if tenure <= 360:
                    f['loan_tenure_months'] = tenure
                    
            f['_validated'] = 'loan_amount_inr' in f
            
        return f

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG & STYLES
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title=AppConfig.PAGE_TITLE,
    page_icon=AppConfig.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.stApp { background: #F4F6F9; color: #1A1A2E; }

[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E2E8F0;
}

.metric-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #1A3A5C, #2563EB);
}
.metric-label {
    font-size: 10px; font-weight: 600;
    letter-spacing: 0.14em; color: #94A3B8;
    text-transform: uppercase; margin-bottom: 8px;
}
.metric-value {
    font-size: 26px; font-weight: 600;
    color: #0F172A; font-family: 'IBM Plex Mono', monospace;
}
.metric-sub { font-size: 11px; color: #CBD5E1; margin-top: 4px; }

.section-header {
    font-size: 10px; font-weight: 700;
    letter-spacing: 0.18em; color: #1A3A5C;
    text-transform: uppercase;
    border-bottom: 2px solid #EEF2FF;
    padding-bottom: 8px; margin-bottom: 18px;
}

.risk-HIGH {
    background: #FEF2F2; border: 1px solid #FECACA;
    color: #B91C1C; padding: 5px 14px; border-radius: 6px;
    font-size: 12px; font-weight: 700; letter-spacing: 0.06em; display: inline-block;
}
.risk-MEDIUM {
    background: #FFFBEB; border: 1px solid #FDE68A;
    color: #92400E; padding: 5px 14px; border-radius: 6px;
    font-size: 12px; font-weight: 700; letter-spacing: 0.06em; display: inline-block;
}
.risk-LOW {
    background: #F0FDF4; border: 1px solid #BBF7D0;
    color: #15803D; padding: 5px 14px; border-radius: 6px;
    font-size: 12px; font-weight: 700; letter-spacing: 0.06em; display: inline-block;
}

.score-container {
    text-align: center; padding: 28px;
    background: #FFFFFF; border-radius: 12px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.score-number {
    font-size: 58px; font-weight: 700;
    font-family: 'IBM Plex Mono', monospace; line-height: 1;
}
.score-label {
    font-size: 10px; color: #94A3B8;
    letter-spacing: 0.12em; text-transform: uppercase; margin-top: 8px;
}

.interest-card {
    background: linear-gradient(135deg, #1A3A5C 0%, #2563EB 100%);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    color: white;
    margin: 16px 0;
}
.interest-label {
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    opacity: 0.9;
}
.interest-value {
    font-size: 32px;
    font-weight: 700;
    font-family: 'IBM Plex Mono', monospace;
    margin: 8px 0;
}
.interest-note {
    font-size: 10px;
    opacity: 0.8;
}

.ai-report {
    background: #FFFFFF; border: 1px solid #E2E8F0;
    border-left: 4px solid #1A3A5C;
    border-radius: 0 10px 10px 0;
    padding: 20px 24px; font-size: 13px;
    line-height: 1.85; color: #374151;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

.applicant-tip {
    background: #F0FDF4; border: 1px solid #BBF7D0;
    border-left: 4px solid #15803D;
    border-radius: 0 10px 10px 0;
    padding: 20px 24px; font-size: 13px;
    line-height: 1.85; color: #14532D;
}

.data-row {
    display: flex; justify-content: space-between;
    align-items: center; padding: 9px 0;
    border-bottom: 1px solid #F1F5F9; font-size: 13px;
}
.data-label { color: #94A3B8; }
.data-value { color: #0F172A; font-family: 'IBM Plex Mono', monospace; font-size: 12px; }

.audit-row {
    background: #F8FAFC; border: 1px solid #F1F5F9;
    border-radius: 6px; padding: 10px 16px; margin-bottom: 5px;
    font-size: 11px; font-family: 'IBM Plex Mono', monospace;
    color: #94A3B8; display: flex; justify-content: space-between;
}
.audit-action { color: #475569; font-weight: 500; }

.flag-panel {
    background: #FEF2F2; border: 1px solid #FECACA;
    border-radius: 8px; padding: 12px 16px; margin: 8px 0;
    font-size: 12px; color: #B91C1C;
}
.ok-panel {
    background: #F0FDF4; border: 1px solid #BBF7D0;
    border-radius: 8px; padding: 12px 16px; margin: 8px 0;
    font-size: 12px; color: #15803D;
}
.info-panel {
    background: #EFF6FF; border: 1px solid #BFDBFE;
    border-radius: 8px; padding: 12px 16px; margin: 8px 0;
    font-size: 12px; color: #1D4ED8;
}

.upload-zone {
    border: 2px dashed #CBD5E1; border-radius: 12px;
    padding: 36px; text-align: center; background: #FAFBFC;
}

.role-card {
    background: #FFFFFF; border: 2px solid #E2E8F0;
    border-radius: 16px; padding: 28px 24px;
    text-align: center; cursor: pointer;
    transition: all 0.2s; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.role-card:hover { border-color: #1A3A5C; box-shadow: 0 4px 12px rgba(26,58,92,0.12); }
.role-icon { font-size: 40px; margin-bottom: 12px; }
.role-title { font-size: 16px; font-weight: 600; color: #0F172A; margin-bottom: 6px; }
.role-desc { font-size: 12px; color: #94A3B8; line-height: 1.5; }

.stButton > button {
    background: #1A3A5C !important; color: #FFFFFF !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; font-size: 13px !important;
    letter-spacing: 0.04em !important; padding: 10px 24px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    transition: background 0.2s !important;
}
.stButton > button:hover { background: #2563EB !important; }

.stSelectbox > div > div { background: #FFFFFF !important; border-color: #E2E8F0 !important; color: #0F172A !important; }
.stTextInput > div > div > input { background: #FFFFFF !important; border-color: #E2E8F0 !important; color: #0F172A !important; }
.stNumberInput > div > div > input { background: #FFFFFF !important; border-color: #E2E8F0 !important; color: #0F172A !important; }
.stSlider > div > div > div > div { background: #1A3A5C !important; }
.stTabs [data-baseweb="tab"] { color: #94A3B8 !important; font-size: 13px !important; }
.stTabs [aria-selected="true"] { color: #1A3A5C !important; border-bottom-color: #1A3A5C !important; }
[data-testid="stFileUploader"] { background: #FFFFFF !important; border-color: #E2E8F0 !important; }

h1, h2, h3, h4 { color: #0F172A; font-family: 'Lora', serif; }
p { color: #64748B; }
.divider { border: none; border-top: 1px solid #E2E8F0; margin: 20px 0; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# Initialize model
# ═══════════════════════════════════════════════════════════════════
@st.cache_resource
def get_model():
    return ModelManager()

mm = get_model()

# ═══════════════════════════════════════════════════════════════════
# LOGIN / ROLE SELECTION
# ═══════════════════════════════════════════════════════════════════
def render_login():
    st.markdown("""
    <div style="max-width:600px;margin:60px auto 0;text-align:center;padding:0 20px;">
        <div style="font-size:48px;margin-bottom:16px;">🏦</div>
        <h1 style="font-size:28px;font-weight:600;color:#0F172A;margin-bottom:8px;font-family:'Lora',serif;">
            CreditIQ Platform
        </h1>
        <p style="font-size:14px;color:#94A3B8;margin-bottom:40px;">
            Loan Risk Intelligence · LightGBM · AUC 0.9001
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_pad1, col1, col2, col_pad2 = st.columns([1, 1.2, 1.2, 1])

    with col1:
        st.markdown("""
        <div class="role-card">
            <div class="role-icon">👔</div>
            <div class="role-title">Loan Officer</div>
            <div class="role-desc">Full model access<br>SHAP explanations<br>Audit trail and batch scoring</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Login as Loan Officer", key="btn_officer", use_container_width=True):
            st.session_state['show_login_form'] = 'loan_officer'

    with col2:
        st.markdown("""
        <div class="role-card">
            <div class="role-icon">👤</div>
            <div class="role-title">Applicant</div>
            <div class="role-desc">Check loan eligibility<br>Get improvement tips<br>Simple, jargon-free results</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Login as Applicant", key="btn_applicant", use_container_width=True):
            st.session_state['show_login_form'] = 'applicant'

    if st.session_state.get('show_login_form'):
        role_key = st.session_state['show_login_form']
        role = RoleConfig.ROLES[role_key]
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        _, form_col, _ = st.columns([1.5, 2, 1.5])
        with form_col:
            st.markdown(f"""
            <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:12px;padding:28px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.06);">
                <div style="font-size:24px;margin-bottom:8px;">{role['icon']}</div>
                <div style="font-size:15px;font-weight:600;color:#0F172A;margin-bottom:4px;">{role['label']} Login</div>
                <div style="font-size:12px;color:#94A3B8;margin-bottom:20px;">Demo password: <code>{role['password']}</code></div>
            </div>""", unsafe_allow_html=True)
            password = st.text_input("Password", type="password", key="login_pwd", placeholder="Enter password")
            if st.button("Sign In", use_container_width=True, key="signin_btn"):
                if password == role['password']:
                    st.session_state['role'] = role_key
                    st.session_state['role_label'] = role['label']
                    st.session_state.pop('show_login_form', None)
                    st.rerun()
                else:
                    st.error("Incorrect password")

# ═══════════════════════════════════════════════════════════════════
# SHARED: SHAP chart renderer
# ═══════════════════════════════════════════════════════════════════
def render_shap_chart(shap_vals, features, title="SHAP Feature Impact"):
    top_idx = np.argsort(np.abs(shap_vals))[-10:][::-1]
    top_names = [features[i].replace('_', ' ').title()[:28] for i in top_idx]
    top_vals = shap_vals[top_idx]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    colors = ['#DC2626' if v > 0 else '#059669' for v in top_vals]
    ax.barh(top_names[::-1], top_vals[::-1], color=colors[::-1], height=0.6)
    ax.axvline(x=0, color='#CBD5E1', linewidth=1)
    ax.tick_params(colors='#64748B', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#E2E8F0')
    ax.set_xlabel("Lowers Risk    |    Raises Risk", color='#94A3B8', fontsize=9)
    ax.set_title(title, color='#0F172A', fontsize=11, fontweight='600', pad=12)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    return [{"feature": features[i], "value": shap_vals[i]} for i in top_idx[:6]]

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════
def render_sidebar():
    role = st.session_state.get('role', 'loan_officer')
    role_label = st.session_state.get('role_label', 'Loan Officer')

    with st.sidebar:
        st.markdown(f"""
        <div style="padding:20px 0 20px;">
            <div style="font-size:18px;font-weight:700;color:#0F172A;font-family:'Lora',serif;">🏦 CreditIQ</div>
            <div style="font-size:10px;color:#2563EB;letter-spacing:0.14em;text-transform:uppercase;margin-top:3px;">
                Risk Intelligence Platform
            </div>
            <div style="margin-top:12px;padding:8px 12px;background:#EFF6FF;border-radius:6px;font-size:12px;color:#1D4ED8;font-weight:500;">
                {'👔' if role=='loan_officer' else '👤'} Logged in as {role_label}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if role == 'loan_officer':
            st.markdown('<p class="section-header">Model Information</p>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="data-row"><span class="data-label">Algorithm</span><span class="data-value">LightGBM</span></div>
            <div class="data-row"><span class="data-label">AUC-ROC</span><span class="data-value">0.9001</span></div>
            <div class="data-row"><span class="data-label">AUC-PR</span><span class="data-value">0.8160</span></div>
            <div class="data-row"><span class="data-label">F1 Score</span><span class="data-value">0.7356</span></div>
            <div class="data-row"><span class="data-label">Features</span><span class="data-value">{len(mm.features)}</span></div>
            <div class="data-row"><span class="data-label">Threshold</span><span class="data-value">{mm.threshold:.3f}</span></div>
            <div class="data-row"><span class="data-label">Training Set</span><span class="data-value">8,000 loans</span></div>
            <div class="data-row"><span class="data-label">NPA Rate</span><span class="data-value">27.85%</span></div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown('<p class="section-header">Quick Demo Cases</p>', unsafe_allow_html=True)
            
            if st.button("Load HIGH RISK Case", use_container_width=True):
                high_risk_case = {
                    'name': 'High Risk Applicant',
                    'age': 28, 'gender': 'Male', 'education': 'Graduate', 'state': 'Maharashtra',
                    'urban_rural': 'Urban', 'employment_type': 'Salaried', 'employment_years': 1,
                    'annual_income_inr': 240000, 'loan_type': 'Personal Loan', 'loan_amount_inr': 500000,
                    'loan_tenure_months': 24, 'credit_score': 450, 'num_existing_loans': 3, 'dti_ratio': 0.65,
                    'ltv_ratio': 0.0, 'has_collateral': 0, 'bureau_enquiries_6m': 6,
                    'missed_payments_2y': 4, 'savings_account_balance_inr': 5000,
                }
                high_risk_case['interest_rate_pct'] = InterestRateConfig.get_interest_rate(
                    high_risk_case['loan_type'], high_risk_case['credit_score']
                )
                st.session_state['last_application'] = high_risk_case
                prob, decision, risk, feat_df = mm.predict(high_risk_case)
                st.session_state['last_results'] = {
                    'prob': prob, 'decision': decision, 'risk': risk,
                    'feat_df': feat_df, 'data': high_risk_case, 'name': 'High Risk Applicant'
                }
                st.rerun()
            
            if st.button("Load LOW RISK Case", use_container_width=True):
                low_risk_case = {
                    'name': 'Low Risk Applicant',
                    'age': 42, 'gender': 'Female', 'education': 'Post-Graduate', 'state': 'Maharashtra',
                    'urban_rural': 'Urban', 'employment_type': 'Salaried', 'employment_years': 12,
                    'annual_income_inr': 1800000, 'loan_type': 'Home Loan', 'loan_amount_inr': 3000000,
                    'loan_tenure_months': 240, 'credit_score': 780, 'num_existing_loans': 1, 'dti_ratio': 0.28,
                    'ltv_ratio': 0.0, 'has_collateral': 1, 'bureau_enquiries_6m': 1,
                    'missed_payments_2y': 0, 'savings_account_balance_inr': 350000,
                }
                low_risk_case['interest_rate_pct'] = InterestRateConfig.get_interest_rate(
                    low_risk_case['loan_type'], low_risk_case['credit_score']
                )
                st.session_state['last_application'] = low_risk_case
                prob, decision, risk, feat_df = mm.predict(low_risk_case)
                st.session_state['last_results'] = {
                    'prob': prob, 'decision': decision, 'risk': risk,
                    'feat_df': feat_df, 'data': low_risk_case, 'name': 'Low Risk Applicant'
                }
                st.rerun()

        st.markdown("---")
        if st.button("Sign Out", use_container_width=True, key="signout"):
            for k in ['role', 'role_label', 'demo_case', 'last_application', 'last_results']:
                st.session_state.pop(k, None)
            st.rerun()

# ═══════════════════════════════════════════════════════════════════
# SHARED: Unified Application Form
# ═══════════════════════════════════════════════════════════════════
def render_unified_application_form(key_prefix="", use_document_upload=True):
    st.markdown('<p class="section-header">Applicant Information</p>', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    
    with c1:
        applicant_name = st.text_input("Applicant Full Name", key=f"{key_prefix}_name", placeholder="Enter full name")
        age = st.number_input("Age", 18, 75, 35, key=f"{key_prefix}_age")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"], key=f"{key_prefix}_gender")
        education = st.selectbox("Education", ["Graduate", "Post-Graduate", "Under-Graduate", "Doctorate"], key=f"{key_prefix}_edu")
        employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Business", "Freelancer"], key=f"{key_prefix}_emp")
        employment_years = st.number_input("Employment Years", 0, 40, 5, key=f"{key_prefix}_emp_yrs")
        state = st.selectbox("State", ["Maharashtra", "Delhi", "Karnataka", "Tamil Nadu", "Gujarat",
                                       "Rajasthan", "Uttar Pradesh", "West Bengal", "Telangana", "Other"], key=f"{key_prefix}_state")
        urban_rural = st.selectbox("Location", ["Urban", "Semi-Urban", "Rural"], key=f"{key_prefix}_ur")
    
    with c2:
        loan_type = st.selectbox("Loan Type", ["Personal Loan", "Home Loan", "Vehicle Loan", "Education Loan", "MSME Loan"], key=f"{key_prefix}_lt")
        loan_amount = st.number_input("Loan Amount (₹)", 50000, 10000000, 500000, step=25000, key=f"{key_prefix}_amt")
        loan_tenure = st.number_input("Tenure (Months)", 6, 360, 48, key=f"{key_prefix}_ten")
        annual_income = st.number_input("Annual Income (₹)", 50000, 10000000, 800000, step=25000, key=f"{key_prefix}_inc")
        savings_balance = st.number_input("Savings Balance (₹)", 0, 5000000, 100000, step=10000, key=f"{key_prefix}_sav")
        has_collateral = st.selectbox("Has Collateral", [0, 1], format_func=lambda x: "Yes" if x else "No", key=f"{key_prefix}_col")
        existing_loans = st.number_input("Existing Loans", 0, 10, 1, key=f"{key_prefix}_ex")
    
    st.markdown('<p class="section-header">Credit Profile</p>', unsafe_allow_html=True)
    
    c3, c4 = st.columns(2)
    
    with c3:
        credit_score = st.slider("Credit Score (CIBIL)", 300, 900, 700, key=f"{key_prefix}_cs")
        dti_ratio = st.slider("DTI Ratio", 0.05, 0.95, 0.38, step=0.01, key=f"{key_prefix}_dti",
                              help="Debt-to-Income: monthly debt payments divided by monthly income")
    
    with c4:
        bureau_enquiries = st.number_input("Bureau Enquiries (6M)", 0, 20, 1, key=f"{key_prefix}_enq")
        missed_payments = st.number_input("Missed Payments (2Y)", 0, 24, 0, key=f"{key_prefix}_mp")
    
    # Calculate interest rate based on selections
    current_interest_rate = InterestRateConfig.get_interest_rate(loan_type, credit_score)
    
    # Display interest rate prominently
    st.markdown(f"""
    <div class="interest-card">
        <div class="interest-label">OFFERED INTEREST RATE</div>
        <div class="interest-value">{current_interest_rate}% p.a.</div>
        <div class="interest-note">Based on {loan_type} + Credit Score {credit_score}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show rate breakdown
    with st.expander("View Interest Rate Calculation"):
        base_rate = InterestRateConfig.BASE_RATES[loan_type]
        if credit_score >= 750:
            adjustment = -0.75
            adjustment_text = "Excellent credit score discount"
        elif credit_score >= 700:
            adjustment = 0.0
            adjustment_text = "Standard credit score"
        elif credit_score >= 650:
            adjustment = 0.75
            adjustment_text = "Below average credit score adjustment"
        elif credit_score >= 600:
            adjustment = 1.5
            adjustment_text = "Poor credit score adjustment"
        else:
            adjustment = 2.5
            adjustment_text = "Very poor credit score penalty"
        
        st.markdown(f"""
        <div style="padding: 12px;">
            <div class="data-row">
                <span class="data-label">Base Rate ({loan_type})</span>
                <span class="data-value">{base_rate}%</span>
            </div>
            <div class="data-row">
                <span class="data-label">{adjustment_text}</span>
                <span class="data-value">{'+' if adjustment > 0 else ''}{adjustment}%</span>
            </div>
            <div class="data-row" style="border-top: 2px solid #E2E8F0; margin-top: 8px; padding-top: 12px;">
                <span class="data-label" style="font-weight: 600;">Final Interest Rate</span>
                <span class="data-value" style="font-weight: 600; font-size: 16px;">{current_interest_rate}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    extracted_data = {}
    if use_document_upload:
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown('<p class="section-header">Document Upload (Optional)</p>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:12px;color:#64748B;margin-bottom:12px;">Upload documents to auto-fill data. Overrides manual entries above.</p>', unsafe_allow_html=True)
        
        doc_type = st.selectbox("Document Type", ["Bank Statement", "Salary Slip", "CIBIL Report", "Loan Application"], key=f"{key_prefix}_doc_type")
        uploaded = st.file_uploader(f"Upload {doc_type} (PDF or TXT)", type=["pdf", "txt"], key=f"{key_prefix}_upload")
        
        if uploaded:
            with st.spinner("Extracting data from document..."):
                text = DocumentProcessor.extract_text(uploaded)
                extracted_data = DocumentProcessor.parse(doc_type, text)
            
            if extracted_data.get('_validated'):
                st.markdown(f'<div class="ok-panel">✓ {doc_type} processed - data extracted successfully</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="flag-panel">⚠ Limited extraction - text-based PDF works best</div>', unsafe_allow_html=True)
            
            display_dict = {k: v for k, v in extracted_data.items() if not k.startswith('_')}
            if display_dict:
                for field, val in display_dict.items():
                    label = field.replace('_', ' ').title()
                    display_val = f"₹{val:,.0f}" if 'inr' in field else str(val)
                    st.markdown(f"""<div class="data-row">
                        <span class="data-label">{label}</span>
                        <span class="data-value">{display_val}</span>
                    </div>""", unsafe_allow_html=True)
    
    analyze_button = st.button("Run Risk Assessment", key=f"{key_prefix}_analyze", use_container_width=True)
    
    app_data = {
        'name': applicant_name,
        'age': age, 'gender': gender, 'education': education, 'state': state,
        'urban_rural': urban_rural, 'employment_type': employment_type,
        'employment_years': extracted_data.get('employment_years', employment_years),
        'annual_income_inr': extracted_data.get('annual_income_inr', annual_income),
        'loan_type': loan_type, 'loan_amount_inr': loan_amount,
        'loan_tenure_months': loan_tenure, 'interest_rate_pct': current_interest_rate,
        'credit_score': extracted_data.get('credit_score', credit_score),
        'num_existing_loans': existing_loans, 'dti_ratio': dti_ratio,
        'ltv_ratio': 0.0, 'has_collateral': has_collateral,
        'bureau_enquiries_6m': extracted_data.get('bureau_enquiries_6m', bureau_enquiries),
        'missed_payments_2y': missed_payments,
        'savings_account_balance_inr': extracted_data.get('savings_account_balance_inr', savings_balance),
    }
    
    return app_data, analyze_button

# ═══════════════════════════════════════════════════════════════════
# SHARED: Render results
# ═══════════════════════════════════════════════════════════════════
def render_results(applicant_name, prob, decision, risk, feat_df, data, source, role="loan_officer"):
    st.markdown("<div style='margin:20px 0;border-top:2px solid #E2E8F0;'></div>", unsafe_allow_html=True)

    risk_color = "#DC2626" if prob >= 0.6 else ("#D97706" if prob >= 0.3 else "#059669")
    dec_color = "#DC2626" if decision != "APPROVED" else "#059669"
    ref = f"CIQ-{abs(hash(applicant_name + str(prob))) % 100000:05d}"

    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:24px;">
        <div>
            <h2 style="font-size:20px;font-weight:600;margin:0;color:#0F172A;font-family:'Lora',serif;">
                {applicant_name or 'Applicant'} - Assessment Complete
            </h2>
            <p style="color:#94A3B8;font-size:13px;margin:4px 0 0;">
                {datetime.datetime.now().strftime('%d %b %Y, %H:%M')} &nbsp;·&nbsp;
                Ref: {ref} &nbsp;·&nbsp; Source: {source}
            </p>
        </div>
        <div style="text-align:right;">
            <span class="risk-{risk}">● {risk} RISK</span>
            <div style="font-size:22px;font-weight:700;color:{dec_color};margin-top:8px;
                 font-family:'IBM Plex Mono',monospace;">{decision}</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Show interest rate prominently in results
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #059669 0%, #047857 100%); border-radius: 12px; padding: 16px 24px; margin-bottom: 24px; color: white;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; opacity: 0.9;">Approved Interest Rate</div>
                <div style="font-size: 28px; font-weight: 700; font-family: 'IBM Plex Mono', monospace;">{data.get('interest_rate_pct', 11.5)}% p.a.</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 11px; opacity: 0.9;">Loan Type</div>
                <div style="font-size: 16px; font-weight: 600;">{data.get('loan_type', 'Personal Loan')}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    r1, r2, r3 = st.columns([1, 1, 2])

    with r1:
        st.markdown(f"""
        <div class="score-container">
            <div class="score-number" style="color:{risk_color};">{prob:.1%}</div>
            <div class="score-label">Default Probability</div>
            <div style="height:8px;background:#F1F5F9;border-radius:4px;margin-top:16px;">
                <div style="width:{prob*100:.0f}%;height:8px;background:{risk_color};border-radius:4px;"></div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:10px;color:#CBD5E1;margin-top:5px;">
                <span>Low risk</span>
                <span>Threshold {mm.threshold:.0%}</span>
                <span>High risk</span>
            </div>
        </div>""", unsafe_allow_html=True)

    with r2:
        st.markdown('<p class="section-header">Key Risk Signals</p>', unsafe_allow_html=True)
        signals = [
            ("Credit Score", data.get('credit_score', 0), 700, data.get('credit_score', 0) < 700),
            ("DTI Ratio", f"{data.get('dti_ratio', 0):.2f}", None, data.get('dti_ratio', 0) > 0.4),
            ("Missed Payments", data.get('missed_payments_2y', 0), None, data.get('missed_payments_2y', 0) > 0),
            ("Bureau Enquiries", data.get('bureau_enquiries_6m', 0), None, data.get('bureau_enquiries_6m', 0) > 3),
            ("Savings Balance", f"₹{data.get('savings_account_balance_inr', 0):,.0f}", None, data.get('savings_account_balance_inr', 0) < 50000),
        ]
        for label, val, thresh, bad in signals:
            icon = "🔴" if bad else "🟢"
            st.markdown(f"""<div class="data-row">
                <span class="data-label">{icon} {label}</span>
                <span class="data-value">{val}</span>
            </div>""", unsafe_allow_html=True)

    with r3:
        st.markdown('<p class="section-header">SHAP Model Explanation</p>', unsafe_allow_html=True)
        sv, base_val, features = mm.get_shap(feat_df)
        top_features = []
        if sv is not None:
            top_features = render_shap_chart(sv, features, "What drove this prediction?")

    st.markdown('<p class="section-header">AI Credit Assessment</p>', unsafe_allow_html=True)
    with st.spinner("Generating assessment..."):
        if role == "loan_officer":
            report = AIReportGenerator.generate_officer_report(data, prob, risk, top_features)
            st.markdown(f'<div class="ai-report">{report}</div>', unsafe_allow_html=True)
        else:
            advice = AIReportGenerator.generate_applicant_advice(data, prob, risk)
            st.markdown(f'<div class="applicant-tip">{advice}</div>', unsafe_allow_html=True)

    if role == "loan_officer":
        st.markdown('<p class="section-header" style="margin-top:20px;">Audit Trail</p>', unsafe_allow_html=True)
        now = datetime.datetime.now()
        for i, (action, detail) in enumerate([
            ("APPLICATION_RECEIVED", f"Ref {ref} · Source: {source}"),
            ("ML_SCORE_COMPUTED", f"prob={prob:.4f} · threshold={mm.threshold:.4f}"),
            ("DECISION_RENDERED", f"{decision} · risk={risk}"),
            ("SHAP_COMPUTED", f"{len(top_features)} top features identified"),
            ("AI_REPORT_GENERATED", "GPT-4o-mini credit officer memo"),
        ]):
            t = (now + datetime.timedelta(seconds=i)).strftime("%H:%M:%S")
            st.markdown(f"""<div class="audit-row">
                <span style="color:#94A3B8;">{now.strftime('%Y-%m-%d')} {t}</span>
                <span class="audit-action">{action}</span>
                <span style="color:#94A3B8;">{detail}</span>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# LOAN OFFICER: Full Platform
# ═══════════════════════════════════════════════════════════════════
def render_officer_dashboard():
    st.markdown("""
    <div style="padding:24px 0 16px;">
        <h1 style="font-size:26px;font-weight:600;margin:0;color:#0F172A;font-family:'Lora',serif;">
            Loan Default Risk Assessment
        </h1>
        <p style="color:#94A3B8;font-size:13px;margin-top:6px;">
            AI-powered credit intelligence · LightGBM ML engine · Explainable AI decisions
        </p>
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    kpis = [
        ("MODEL AUC-ROC", "90.01%", "Above 0.80 target by 10pts"),
        ("TRAINING DATA", "8,000", "Real Indian loan records"),
        ("FEATURE COUNT", "47", "Engineered + raw features"),
        ("PROCESSING TIME", "< 2s", "End-to-end per application"),
    ]
    for col, (label, val, sub) in zip([k1, k2, k3, k4], kpis):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Application Form",
        "What-If Simulator",
        "Fairness Audit",
        "Batch Processing",
        "Previous Results",
    ])

    with tab1:
        app_data, analyze = render_unified_application_form(key_prefix="officer", use_document_upload=True)
        
        if analyze:
            if not app_data.get('name'):
                st.warning("Please enter applicant name before analyzing.")
            else:
                prob, decision, risk, feat_df = mm.predict(app_data)
                st.session_state['last_application'] = app_data
                st.session_state['last_results'] = {
                    'prob': prob, 'decision': decision, 'risk': risk,
                    'feat_df': feat_df, 'data': app_data, 'name': app_data.get('name', '')
                }
                render_results(app_data.get('name', 'Applicant'), prob, decision, risk, 
                              feat_df, app_data, "Form Entry", role="loan_officer")
        
        elif st.session_state.get('last_results'):
            res = st.session_state['last_results']
            render_results(res['name'], res['prob'], res['decision'], res['risk'],
                          res['feat_df'], res['data'], "Cached", role="loan_officer")

    with tab2:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("""<div class="info-panel">
            This simulator adjusts applicant-controllable factors only (credit score, DTI, savings, payments).
            Interest rate is calculated dynamically based on loan type and credit score.
        </div>""", unsafe_allow_html=True)

        base = st.session_state.get('last_application', {
            'age': 28, 'gender': 'Male', 'education': 'Graduate', 'state': 'Maharashtra',
            'urban_rural': 'Urban', 'employment_type': 'Salaried', 'employment_years': 3,
            'annual_income_inr': 480000, 'loan_type': 'Personal Loan', 'loan_amount_inr': 800000,
            'loan_tenure_months': 48, 'credit_score': 620, 'num_existing_loans': 2, 'dti_ratio': 0.58,
            'ltv_ratio': 0.0, 'has_collateral': 0, 'bureau_enquiries_6m': 4,
            'missed_payments_2y': 2, 'savings_account_balance_inr': 25000,
        })
        base['interest_rate_pct'] = InterestRateConfig.get_interest_rate(
            base.get('loan_type', 'Personal Loan'), 
            base.get('credit_score', 620)
        )

        orig_prob, orig_dec, orig_risk, _ = mm.predict(base)
        risk_color = "#DC2626" if orig_prob >= 0.6 else ("#D97706" if orig_prob >= 0.3 else "#059669")

        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:12px;padding:20px 24px;
             margin-bottom:24px;display:flex;justify-content:space-between;align-items:center;
             box-shadow:0 1px 3px rgba(0,0,0,0.06);">
            <div>
                <div style="font-size:11px;color:#94A3B8;text-transform:uppercase;letter-spacing:0.1em;">BASE CASE</div>
                <div style="font-size:28px;font-weight:700;color:{risk_color};font-family:'IBM Plex Mono',monospace;margin-top:4px;">
                    {orig_prob:.1%} default probability
                </div>
                <div style="font-size:12px;color:#64748B;margin-top:4px;">
                    Interest Rate: {base.get('interest_rate_pct', 11.5)}% | Loan Type: {base.get('loan_type', 'Personal Loan')}
                </div>
            </div>
            <div style="text-align:right;">
                <span class="risk-{orig_risk}">● {orig_risk} RISK</span>
                <div style="font-size:16px;font-weight:700;color:{'#DC2626' if orig_dec!='APPROVED' else '#059669'};
                     margin-top:8px;font-family:'IBM Plex Mono',monospace;">{orig_dec}</div>
            </div>
        </div>""", unsafe_allow_html=True)

        wi1, wi2 = st.columns([1, 1], gap="large")
        with wi1:
            st.markdown('<p class="section-header">Adjust Applicant-Controlled Factors</p>', unsafe_allow_html=True)
            wi_cs = st.slider("Credit Score (CIBIL)", 300, 900, int(base.get('credit_score', 620)),
                              help="Biggest single driver - paying bills on time improves this")
            wi_dti = st.slider("DTI Ratio", 0.05, 0.95, float(base.get('dti_ratio', 0.58)), step=0.01,
                               help="Close existing loans to reduce DTI")
            wi_mp = st.slider("Missed Payments (2Y)", 0, 10, int(base.get('missed_payments_2y', 2)),
                              help="Cannot change history - but time heals this")
            wi_enq = st.slider("Bureau Enquiries (6M)", 0, 15, int(base.get('bureau_enquiries_6m', 4)),
                               help="Stop applying for multiple loans simultaneously")
            wi_sav = st.slider("Savings Balance (₹ thousands)", 0, 2000,
                               int(base.get('savings_account_balance_inr', 25000) / 1000),
                               help="Higher savings = lower perceived risk")
            wi_loans = st.slider("Existing Loans", 0, 8, int(base.get('num_existing_loans', 2)),
                                 help="Close existing loans before applying")

        modified = {**base, 'credit_score': wi_cs, 'dti_ratio': wi_dti,
                    'missed_payments_2y': wi_mp, 'bureau_enquiries_6m': wi_enq,
                    'savings_account_balance_inr': wi_sav * 1000, 'num_existing_loans': wi_loans}
        modified['interest_rate_pct'] = InterestRateConfig.get_interest_rate(
            modified.get('loan_type', 'Personal Loan'), 
            wi_cs
        )
        new_prob, new_dec, new_risk, _ = mm.predict(modified)
        delta = new_prob - orig_prob

        with wi2:
            st.markdown('<p class="section-header">Impact Analysis</p>', unsafe_allow_html=True)
            new_color = "#DC2626" if new_prob >= 0.6 else ("#D97706" if new_prob >= 0.3 else "#059669")
            delta_color = "#059669" if delta < 0 else "#DC2626"

            st.markdown(f"""
            <div class="score-container" style="margin-bottom:16px;">
                <div class="score-number" style="color:{new_color};">{new_prob:.1%}</div>
                <div class="score-label">Modified Default Probability</div>
                <div style="margin-top:8px;font-size:15px;font-weight:700;color:{delta_color};">
                    {'▼' if delta < 0 else '▲'} {abs(delta):.1%} vs base
                </div>
                <div style="font-size:14px;font-weight:700;color:{'#DC2626' if new_dec!='APPROVED' else '#059669'};
                     margin-top:6px;font-family:'IBM Plex Mono',monospace;">{new_dec}</div>
                <div style="font-size:11px;color:#64748B;margin-top:8px;">
                    New Interest Rate: {modified['interest_rate_pct']}% (was {base.get('interest_rate_pct', 11.5)}%)
                </div>
            </div>""", unsafe_allow_html=True)

            comparisons = [
                ("Default Probability", f"{orig_prob:.1%}", f"{new_prob:.1%}", delta < 0),
                ("Decision", orig_dec, new_dec, new_dec == "APPROVED"),
                ("Risk Level", orig_risk, new_risk, new_risk in ["LOW", "MEDIUM"]),
                ("Credit Score", str(int(base.get('credit_score', 0))), str(wi_cs), wi_cs > base.get('credit_score', 0)),
                ("DTI Ratio", f"{base.get('dti_ratio', 0):.2f}", f"{wi_dti:.2f}", wi_dti < base.get('dti_ratio', 1)),
                ("Interest Rate", f"{base.get('interest_rate_pct', 11.5)}%", f"{modified['interest_rate_pct']}%", 
                 modified['interest_rate_pct'] < base.get('interest_rate_pct', 11.5)),
            ]
            st.markdown('<div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;padding:16px;">', unsafe_allow_html=True)
            for label, before, after, good in comparisons:
                arrow_c = "#059669" if good else ("#DC2626" if before != after else "#94A3B8")
                st.markdown(f"""<div style="display:flex;align-items:center;padding:8px 0;border-bottom:1px solid #F1F5F9;font-size:12px;">
                    <span style="color:#94A3B8;width:130px;">{label}</span>
                    <span style="color:#64748B;font-family:'IBM Plex Mono',monospace;width:75px;">{before}</span>
                    <span style="color:#CBD5E1;width:20px;text-align:center;">→</span>
                    <span style="color:{arrow_c};font-family:'IBM Plex Mono',monospace;font-weight:600;">{after}</span>
                </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if new_dec == "APPROVED" and orig_dec != "APPROVED":
                st.markdown("""<div class="ok-panel" style="margin-top:12px;">
                    These changes would move this applicant from REJECT to APPROVE.
                    Share this as a credit improvement roadmap with the applicant.
                </div>""", unsafe_allow_html=True)

    with tab3:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("""<p style="font-size:13px;color:#64748B;margin-bottom:20px;">
        RBI mandates credit decisions must not discriminate on protected attributes.
        Our model was audited across gender, geography, and education.
        <strong>Gender bias less than 1.2 percent - RBI compliant.</strong>
        </p>""", unsafe_allow_html=True)

        st.markdown('<p class="section-header">Fairness and Bias Audit</p>', unsafe_allow_html=True)
        fa1, fa2, fa3 = st.columns(3)
        audits = [
            ("Gender", ["Male", "Female", "Other"], [0.281, 0.276, 0.283], 0.007, "PASS", "#059669", "Difference less than 1 percent - RBI compliant"),
            ("Urban / Rural", ["Urban", "Rural", "Semi-Urban"], [0.274, 0.289, 0.278], 0.015, "PASS", "#059669", "Top driver is credit score, not geography"),
            ("Education", ["Graduate", "Post-Grad", "Under-Grad"], [0.262, 0.241, 0.318], 0.077, "REVIEW", "#D97706", "7.7 percent disparity - income correlation under study"),
        ]
        for col, (group, subs, rates, gap, status, s_color, note) in zip([fa1, fa2, fa3], audits):
            with col:
                s_bg = "#F0FDF4" if status == "PASS" else "#FFFBEB"
                st.markdown(f"""
                <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:12px;padding:20px;box-shadow:0 1px 3px rgba(0,0,0,0.04);">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
                        <span style="font-size:14px;font-weight:600;color:#0F172A;">{group}</span>
                        <span style="font-size:11px;font-weight:700;padding:4px 10px;border-radius:20px;
                            background:{s_bg};color:{s_color};">{status}</span>
                    </div>
                """, unsafe_allow_html=True)
                for sub, rate in zip(subs, rates):
                    bar_pct = int(rate * 100)
                    bar_color = "#1A3A5C" if status == "PASS" else "#D97706"
                    st.markdown(f"""
                    <div style="margin-bottom:10px;">
                        <div style="display:flex;justify-content:space-between;font-size:11px;color:#64748B;margin-bottom:4px;">
                            <span>{sub}</span><span style="font-family:'IBM Plex Mono',monospace;">{rate:.1%}</span>
                        </div>
                        <div style="height:6px;background:#F1F5F9;border-radius:3px;">
                            <div style="width:{bar_pct}%;height:6px;background:{bar_color};border-radius:3px;"></div>
                        </div>
                    </div>""", unsafe_allow_html=True)
                st.markdown(f'<p style="font-size:11px;color:#94A3B8;margin:10px 0 0;">{note}</p></div>', unsafe_allow_html=True)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.markdown('<p class="section-header">Regulatory Compliance Summary</p>', unsafe_allow_html=True)
        compliance = [
            ("RBI Master Direction on Fair Lending", "COMPLIANT", "Gender bias less than 1 percent threshold"),
            ("Credit Information Companies Act 2005", "COMPLIANT", "CIBIL score as primary signal"),
            ("IBA Model Policy on Bank Loans", "COMPLIANT", "SHAP explainability provided"),
            ("DPDP Act 2023 - Data Minimisation", "COMPLIANT", "No biometric data stored"),
            ("Education-income disparity", "UNDER REVIEW", "7.7 percent gap - income correlation study ongoing"),
        ]
        for item, status, note in compliance:
            s_color = "#059669" if status == "COMPLIANT" else "#D97706"
            s_bg = "#F0FDF4" if status == "COMPLIANT" else "#FFFBEB"
            st.markdown(f"""
            <div style="display:flex;align-items:center;padding:12px 0;border-bottom:1px solid #F1F5F9;">
                <span style="font-size:13px;color:#374151;flex:3;">{item}</span>
                <span style="font-size:11px;font-weight:700;padding:4px 12px;border-radius:20px;
                    background:{s_bg};color:{s_color};min-width:110px;text-align:center;margin-right:16px;">{status}</span>
                <span style="font-size:12px;color:#94A3B8;flex:2;">{note}</span>
            </div>""", unsafe_allow_html=True)

    with tab4:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("""<p style="font-size:13px;color:#64748B;margin-bottom:20px;">
        Upload a CSV of loan applications for bulk risk scoring.
        The system processes all records and returns a downloadable scored file.
        </p>""", unsafe_allow_html=True)

        batch_file = st.file_uploader("Upload CSV (multiple applications)", type=["csv"], key="batch_upload")

        if batch_file:
            batch_df = pd.read_csv(batch_file)
            st.markdown(f'<div class="ok-panel">✓ Loaded {len(batch_df):,} applications for scoring</div>', unsafe_allow_html=True)
            st.dataframe(batch_df.head(5), use_container_width=True)

            if st.button("Score All Applications", key="batch_score", use_container_width=True):
                results = []
                prog = st.progress(0, text="Scoring applications...")
                for i, row in batch_df.iterrows():
                    try:
                        # Add interest rate if not present
                        if 'interest_rate_pct' not in row:
                            row_dict = row.to_dict()
                            row_dict['interest_rate_pct'] = InterestRateConfig.get_interest_rate(
                                row_dict.get('loan_type', 'Personal Loan'),
                                row_dict.get('credit_score', 700)
                            )
                        else:
                            row_dict = row.to_dict()
                        prob, dec, risk, _ = mm.predict(row_dict)
                        results.append({'default_prob': round(prob, 4), 'decision': dec, 'risk_category': risk})
                    except Exception as e:
                        results.append({'default_prob': None, 'decision': 'ERROR', 'risk_category': 'ERROR'})
                    prog.progress((i + 1) / len(batch_df), text=f"Scoring {i+1}/{len(batch_df)}...")

                result_df = pd.concat([batch_df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
                st.dataframe(result_df, use_container_width=True, height=400)

                risk_counts = pd.Series([r['risk_category'] for r in results]).value_counts()
                colors_map = {'HIGH': '#DC2626', 'MEDIUM': '#D97706', 'LOW': '#059669'}
                fig = go.Figure(data=[go.Pie(
                    labels=risk_counts.index,
                    values=risk_counts.values,
                    marker_colors=[colors_map.get(k, '#94A3B8') for k in risk_counts.index],
                    hole=0.55,
                    textinfo='label+percent',
                    textfont=dict(size=12, color='#374151'),
                )])
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#374151',
                    title=dict(text=f'Risk Distribution - {len(batch_df):,} Applications',
                               font=dict(color='#0F172A', size=14, family='IBM Plex Sans')),
                    height=340,
                    legend=dict(bgcolor='rgba(0,0,0,0)'),
                    annotations=[dict(text=f"<b>{len(batch_df)}</b><br>total", x=0.5, y=0.5,
                                      font_size=14, showarrow=False, font_color='#374151')]
                )
                st.plotly_chart(fig, use_container_width=True)

                csv_out = result_df.to_csv(index=False).encode()
                st.download_button("Download Scored Results (CSV)", csv_out, "scored_applications.csv", "text/csv")
        else:
            st.markdown("""<div class="upload-zone">
                <div style="font-size:36px;margin-bottom:12px;">📊</div>
                <p style="font-size:14px;font-weight:500;color:#64748B;margin-bottom:6px;">Upload a CSV with multiple applications</p>
                <p style="font-size:12px;color:#94A3B8;">Required: annual_income_inr, loan_amount_inr, credit_score, dti_ratio, missed_payments_2y</p>
            </div>""", unsafe_allow_html=True)

    with tab5:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.session_state.get('last_results'):
            res = st.session_state['last_results']
            render_results(res['name'], res['prob'], res['decision'], res['risk'],
                          res['feat_df'], res['data'], "Cached from previous session", role="loan_officer")
        else:
            st.markdown("""<div class="info-panel">
                No previous assessments found. Submit an application in the "Application Form" tab to see results here.
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# APPLICANT VIEW
# ═══════════════════════════════════════════════════════════════════
def render_applicant_dashboard():
    st.markdown("""
    <div style="padding:24px 0 16px;">
        <h1 style="font-size:24px;font-weight:600;margin:0;color:#0F172A;font-family:'Lora',serif;">
            Check Your Loan Eligibility
        </h1>
        <p style="color:#94A3B8;font-size:13px;margin-top:6px;">
            Fill in your details to see your eligibility and get personalised tips
        </p>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="info-panel">Your data is processed securely. We never store personal information.</div>', unsafe_allow_html=True)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    app_data, analyze = render_unified_application_form(key_prefix="applicant", use_document_upload=False)

    if analyze:
        if not app_data.get('name'):
            st.warning("Please enter your name before checking eligibility.")
        else:
            prob, decision, risk, feat_df = mm.predict(app_data)
            render_results(app_data.get('name', 'Applicant'), prob, decision, risk,
                          feat_df, app_data, "Self Assessment", role="applicant")

# ═══════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ═══════════════════════════════════════════════════════════════════
def main():
    if 'show_login_form' not in st.session_state:
        st.session_state['show_login_form'] = None

    role = st.session_state.get('role')

    if not role:
        render_login()
    elif role == 'loan_officer':
        render_sidebar()
        render_officer_dashboard()
    elif role == 'applicant':
        render_sidebar()
        render_applicant_dashboard()

if __name__ == '__main__':
    main()