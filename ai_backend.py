from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from groq import Groq
import json
from typing import Dict, List, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from contextlib import asynccontextmanager
import urllib.parse
import uvicorn
import os
import logging
import time
import logging
logger = logging.getLogger(__name__)
from fastapi.middleware.cors import CORSMiddleware
try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    logger.warning(f"Failed to import SentenceTransformer: {str(e)}. Embeddings will be skipped.")
    SentenceTransformer = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Groq client
client = Groq(api_key="your api key")
df = None
model_pipeline = None
audit_policies = None
recommendations_cache = None
embeddings = None

# Define features
numerical_features = [
    'Final_Total', 'Discount', 'Tax', 'Sub_Total', 'CGST_Amount',
    'SGST_Amount', 'VAT_Amount', 'Service_Charge_Amount',
    'Online_Tax_Calculated_s', 'Online_Tax_Calculated_z',
    'Cancelled_Invoice_Total_co', 'Food_Preparation_Time_Z',
    'Food_Preparation_Time_S', 'Is_Online'
]
categorical_features = [
    'Order_Type', 'Payment_Type', 'Status', 'Order_Source'
]

# Load audit policies
try:
    with open('audit_policies.json') as f:
        AUDIT_POLICIES = json.load(f)['audit_policies']
except FileNotFoundError:
    logger.warning("❌ 'audit_policies.json' not found. Using default policies.")
    AUDIT_POLICIES = [
        {
            "policy_name": "Incorrect GST Calculation",
            "description": "GST calculation error for local orders",
            "condition": "Order_Source == 'Local' and abs(CGST_Amount + SGST_Amount - 0.05 * Sub_Total) > 1",
            "action": "Review tax configuration",
            "severity_range": [0.4, 0.6]
        }
    ]

# Cache file
CACHE_FILE = "recommendations_cache.json"

def convert_value(value):
    if pd.isna(value):
        return None
    elif isinstance(value, (np.floating, float)):
        return float(value)
    elif isinstance(value, (np.integer, int)):
        return int(value)
    elif isinstance(value, (np.bool_, bool)):
        return bool(value)
    else:
        return str(value) if value is not None else None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global df, model_pipeline, audit_policies, recommendations_cache, embeddings
    try:
        logger.info("Starting lifespan setup...")
        df = pd.read_csv(
            "data.csv",
            parse_dates=['Date', 'Timestamp', 'Cancelled_Time_z', 'Cancelled_Time_s'],
            dtype={
                'Final_Total': 'float32',
                'Discount': 'float32',
                'Order_Type': 'category',
                'Payment_Type': 'category',
                'Status': 'category'
            }
        )
        df = df[:5000]  # Adjust for testing
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        
        # Fill missing values
        df.fillna({
            'Discount': 0, 'Final_Total': 0, 'Tax': 0, 'Sub_Total': 0,
            'CGST_Amount': 0, 'SGST_Amount': 0, 'VAT_Amount': 0,
            'Service_Charge_Amount': 0, 'Online_Tax_Calculated_s': 0,
            'Online_Tax_Calculated_z': 0, 'Cancelled_Invoice_Total_co': 0,
            'Food_Preparation_Time_Z': 0, 'Food_Preparation_Time_S': 0
        }, inplace=True)
        
        # Derive features
        df['Is_Online'] = (df['Order_From_z'].notna() | df['Order_From_s'].notna()).astype(int)
        df['Order_Source'] = np.where(
            df['Order_From_z'].notna(), 'Zomato',
            np.where(df['Order_From_s'].notna(), 'Swiggy', 'Local'))
        logger.info("Initialized 'Order_Source' column")

        # Define and fit the pipeline once
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), numerical_features),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore'))
                ]), categorical_features)
            ])

        model_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('detector', IsolationForest(
                contamination=0.2,
                random_state=42,
                n_estimators=20
            ))
        ])
        
        model_pipeline.fit(df)
        logger.info("Model pipeline fitted successfully")

        if SentenceTransformer:
            try:
                logger.info("Generating embeddings...")
                embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
                df['text_for_embedding'] = df.apply(
                    lambda row: f"Order Type: {row['Order_Type']}, Payment: {row['Payment_Type']}, "
                                f"Status: {row['Status']}, Source: {row['Order_Source']}, "
                                f"Total: {row['Final_Total']}, Tax: {row['Tax']}", axis=1
                )
                embeddings = embedder.encode(df['text_for_embedding'].tolist(), show_progress_bar=True)
                df['embeddings'] = [emb.tolist() for emb in embeddings]
                logger.info("Embeddings generated and stored")
            except Exception as e:
                logger.error(f"Failed to generate embeddings: {str(e)}")
                embeddings = None
        else:
            logger.warning("SentenceTransformer not available; skipping embeddings.")
            embeddings = None

        audit_policies = AUDIT_POLICIES
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                recommendations_cache = json.load(f)
            logger.info("✅ Loaded cached recommendations")
        else:
            recommendations_cache = {}
            logger.info("Initialized empty recommendations cache")

        logger.info("✅ Data loaded, model initialized, and policies set")
    except Exception as e:
        logger.error(f"❌ Error during startup: {str(e)}")
        raise
    yield
    if recommendations_cache:
        with open(CACHE_FILE, 'w') as f:
            json.dump(recommendations_cache, f)
        logger.info("✅ Saved recommendations cache")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

class FilterRequest(BaseModel):
    filters: Optional[Dict[str, List[str]]] = None
    severity: Optional[List[str]] = None

class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 500
    temperature: float = 0.7

@app.get("/filters/options")
async def get_filter_options():
    return {
        "Order_Type": df['Order_Type'].cat.categories.tolist(),
        "Payment_Type": df['Payment_Type'].cat.categories.tolist(),
        "Order_Source": ["Zomato", "Swiggy", "Local"],
        "Status": df['Status'].cat.categories.tolist(),
        "Severity": ["high", "medium", "low"]
    }

def apply_filters(data: pd.DataFrame, filters: Dict[str, List[str]]) -> pd.DataFrame:
    if not filters:
        return data.copy()
    mask = pd.Series(True, index=data.index)
    for col, values in filters.items():
        if col in data.columns:
            mask &= data[col].isin(values)
    return data[mask].copy()

def compute_severity_score(row, policy_severity=None):
    w_financial = 0.5
    w_operational = 0.3
    w_statistical = 0.2

    financial_score = min(row.get('Final_Total', 0) / 1000, 1.0)
    if row.get('Status', '') == 'Cancelled':
        financial_score = max(financial_score, min(row.get('Cancelled_Invoice_Total_co', 0) / 1000, 1.0))

    prep_time = max(row.get('Food_Preparation_Time_Z', 0), row.get('Food_Preparation_Time_S', 0))
    operational_score = min(prep_time / 60, 1.0)

    statistical_score = 0.0
    if 'model_anomaly_score' in row and pd.notna(row['model_anomaly_score']):
        statistical_score = min(max(-row['model_anomaly_score'] / 0.2, 0), 1.0)

    rule_boost = 0
    if row.get('detected_flags', ''):
        rule_boost = min(len(row['detected_flags'].split('|')), 3) * 0.1

    policy_factor = np.mean(policy_severity) if policy_severity else 0.5
    total_score = (w_financial * financial_score + 
                   w_operational * operational_score + 
                   w_statistical * statistical_score + rule_boost) * policy_factor
    return min(max(total_score * 100, 0), 100)

def run_anomaly_detection(data: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Starting anomaly detection on {len(data)} rows")
    start_time = time.time()
    
    df_local = data.copy()
    df_local['detected_flags'] = ""
    df_local['model_flags'] = ""
    df_local['severity'] = "medium"
    df_local['severity_score'] = 0.0
    df_local['retrieved_policies'] = ""

    # Vectorized rule-based detection
    logger.info("Applying rule-based detection...")
    
    # Define conditions as tuples (mask, flag)
    conditions = [
        ((df_local['Order_Source'] == 'Local') & 
         (np.abs(df_local['CGST_Amount'] + df_local['SGST_Amount'] - 0.05 * df_local['Sub_Total']) > 1),
         "Incorrect GST Calculation"),
        
        ((df_local['Is_Online'] == 1) & (df_local['Service_Charge_Amount'] > 0),
         "Invalid Service Charge for Online Order"),
        
        (('Non_Taxable' in df_local.columns) & 
         (df_local['Non_Taxable'] == 1) & 
         ((df_local['CGST_Amount'] + df_local['SGST_Amount'] + df_local['VAT_Amount']) > 0),
         "Tax Applied to Non-Taxable Item"),
        
        (('Aggregator_Order_No_z' in df_local.columns) & ('Aggregator_Order_No_s' in df_local.columns) & 
         (df_local['Order_Source'] == 'Local') & 
         (df_local['Aggregator_Order_No_z'].notna() | df_local['Aggregator_Order_No_s'].notna()),
         "Local Order with Aggregator Data"),
        
        ((df_local['Is_Online'] == 1) & 
         (~df_local['Payment_Type'].isin(['Online', 'Other [ZOMATO PAY]', 'Other [Paytm]'])),
         "Invalid Payment Method for Online Order"),
        
        ((df_local['Status'].str.contains('Cancelled', na=False)) & 
         ((df_local['Food_Preparation_Time_Z'] > 30) | (df_local['Food_Preparation_Time_S'] > 30)),
         "Late Cancellation"),
        
        ((df_local['Status'].str.contains('Cancelled', na=False)) & 
         (df_local['Cancelled_Invoice_Total_co'] > 1000),
         "High-Value Cancellation"),
        
        (('Assign_To' in df_local.columns) & 
         (df_local['Status'] == 'Complimentary') & (df_local['Assign_To'].isna()),
         "Unapproved Complimentary Order")
    ]

    # Apply conditions vectorized
    for mask, flag in conditions:
        df_local.loc[mask, 'detected_flags'] += flag + "|"
        df_local.loc[mask, 'retrieved_policies'] += f"{flag}: {next((p['description'] for p in audit_policies if p['policy_name'] == flag), 'No description')}|"

    logger.info(f"Rule-based detection completed in {time.time() - start_time:.2f} seconds")

    # Model-based detection
    if model_pipeline is not None and not data[numerical_features].dropna().empty:
        logger.info("Applying pre-fitted IsolationForest model...")
        scores = model_pipeline.decision_function(df_local)
        predictions = model_pipeline.predict(df_local)
        df_local['model_anomaly_score'] = scores
        df_local.loc[predictions == -1, 'model_flags'] += "Model Detected Anomaly|"
        logger.info(f"Model detection completed in {time.time() - start_time:.2f} seconds")

    df_local['detected_flags'] = df_local['detected_flags'].str.rstrip('|')
    df_local['model_flags'] = df_local['model_flags'].str.rstrip('|')
    df_local['retrieved_policies'] = df_local['retrieved_policies'].str.rstrip('|')
    df_local.loc[df_local['retrieved_policies'] == "", 'retrieved_policies'] = "No relevant policies violated"

    flagged_df = df_local[(df_local['detected_flags'] != "") | (df_local['model_flags'] != "")]
    if len(flagged_df) > 0:
        logger.info("Computing severity scores...")
        flagged_df['severity_score'] = flagged_df.apply(
            lambda row: compute_severity_score(row, [np.mean(p['severity_range']) for p in audit_policies if any(p['policy_name'] in row['detected_flags'] for p in audit_policies)]), 
            axis=1
        )
        flagged_df['severity'] = pd.cut(
            flagged_df['severity_score'],
            bins=[0, 33, 66, 100],
            labels=['low', 'medium', 'high'],
            include_lowest=True
        )
        df_local.loc[flagged_df.index, 'severity_score'] = flagged_df['severity_score']
        df_local.loc[flagged_df.index, 'severity'] = flagged_df['severity']

    logger.info(f"Anomaly detection completed. Flagged {len(flagged_df)} anomalies in {time.time() - start_time:.2f} seconds")
    return flagged_df

@app.post("/audit-report")
async def generate_audit_report(filter_request: FilterRequest):
    global recommendations_cache
    try:
        filtered_df = apply_filters(df, filter_request.filters) if filter_request.filters else df.copy()
        logger.info(f"Filtered {len(filtered_df)} rows")
        flagged_df = run_anomaly_detection(filtered_df)
        logger.info(f"Flagged {len(flagged_df)} anomalies")
        
        report = {
            "anomaly_categories": {},
            "visualization_data": {
                "anomaly_types": {},
                "severity_distribution": {"high": 0, "medium": 0, "low": 0},
                "source_distribution": {}
            },
            "total_stats": {
                "total_records": len(filtered_df),
                "total_anomalies": len(flagged_df),
                "financial_impact": float(flagged_df['Final_Total'].sum())
            }
        }
        
        policy_map = {
            "Incorrect GST Calculation": "incorrect_gst_calculation",
            "Invalid Service Charge for Online Order": "invalid_service_charge",
            "Tax Applied to Non-Taxable Item": "non_taxable_with_tax",
            "Invalid Payment Method for Online Order": "invalid_payment_method",
            "Local Order with Aggregator Data": "local_with_aggregator",
            "Late Cancellation": "late_cancellation",
            "High-Value Cancellation": "high_value_cancellation",
            "Unapproved Complimentary Order": "unapproved_complimentary",
            "Model Detected Anomaly": "model_detected_anomaly"
        }

        for _, row in flagged_df.iterrows():
            anomalies = (row['detected_flags'] + "|" + row['model_flags']).split('|')
            for anomaly in anomalies:
                if not anomaly.strip():
                    continue
                    
                category = policy_map.get(anomaly.strip())
                if category:
                    if category not in report["anomaly_categories"]:
                        if "Model" in anomaly:
                            policy = {
                                "policy_name": "Machine Learning Detected Anomaly",
                                "description": "Anomaly detected by AI model based on complex patterns",
                                "severity": row['severity'],
                                "action": "Review transaction for unusual patterns"
                            }
                        else:
                            policy = next(
                                (p for p in audit_policies if anomaly.split(' (')[0] in p['policy_name']),
                                {"policy_name": anomaly, "description": "RAG-detected anomaly", "action": "Review"}
                            )
                        
                        report["anomaly_categories"][category] = {
                            "records": [],
                            "count": 0,
                            "severity": row['severity'],
                            "policy_details": policy
                        }
                    
                    sample = {col: convert_value(val) for col, val in row.items() if col != 'embeddings'}
                    sample['severity_score'] = float(row['severity_score'])
                    sample['retrieved_policies'] = row['retrieved_policies']
                    report["anomaly_categories"][category]["records"].append(sample)
                    report["anomaly_categories"][category]["count"] += 1

        for category, data in report["anomaly_categories"].items():
            policy_name = data["policy_details"].get("policy_name", "Unknown")
            report["visualization_data"]["anomaly_types"][policy_name] = data["count"]
            report["visualization_data"]["severity_distribution"][data["severity"]] += data["count"]
            source_counts = pd.DataFrame(data["records"])['Order_Source'].value_counts().to_dict()
            for source, count in source_counts.items():
                report["visualization_data"]["source_distribution"][source] = \
                    report["visualization_data"]["source_distribution"].get(source, 0) + count

        for category, data in report["anomaly_categories"].items():
            if data["count"] > 0:
                if category in recommendations_cache:
                    data["recommendations"] = recommendations_cache[category]
                else:
                    policy = data["policy_details"]
                    if "Model" in policy.get("policy_name", ""):
                        prompt = f"""
                        Analyze these {data['count']} machine learning-detected anomalies in restaurant transactions.
                        Common patterns in these anomalies:
                        {json.dumps(pd.DataFrame(data['records']).describe(), indent=2)}
                        
                        Provide:
                        1. Root cause analysis
                        2. 3 specific recommendations
                        3. Potential business impact
                        """
                    else:
                        example = next(
                            (r for r in data["records"] if all(k in r for k in policy.get("relevant_columns", []))),
                            {}
                        )
                        prompt = f"""
                        Generate improvement recommendations for {policy['policy_name']}:
                        - Description: {policy['description']}
                        - Severity: {data['severity']}
                        - Example Case: {json.dumps(example, indent=2)}
                        - Suggested Action: {policy['action']}
                        - Retrieved Policies: {example.get('retrieved_policies', 'None')}
                        
                        Provide:
                        1. Root cause analysis
                        2. 3 specific improvement steps
                        3. Expected business impact
                        """
                    try:
                        response = client.chat.completions.create(
                            messages=[
                                {"role": "system", "content": "You are a financial audit expert."},
                                {"role": "user", "content": prompt}
                            ],
                            model="llama3-70b-8192",
                            temperature=0.3,
                            max_tokens=500
                        )
                        recommendations = response.choices[0].message.content
                        recommendations_cache[category] = recommendations
                        data["recommendations"] = recommendations
                    except Exception as e:
                        recommendations_cache[category] = f"Recommendations unavailable due to API error: {str(e)}"
                        data["recommendations"] = recommendations_cache[category]

        return report
    except Exception as e:
        logger.error(f"Audit report failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect-anomalies")
async def detect_anomalies(filter_request: FilterRequest):
    try:
        filtered_df = apply_filters(df, filter_request.filters) if filter_request.filters else df.copy()
        flagged_df = run_anomaly_detection(filtered_df)
        
        if filter_request.severity:
            flagged_df = flagged_df[flagged_df['severity'].isin(filter_request.severity)]
        
        samples = []
        for _, row in flagged_df.sort_values('severity_score', ascending=False).head(1000).iterrows():
            sample = {col: convert_value(val) for col, val in row.items() if col != 'embeddings'}
            sample['severity_score'] = float(row['severity_score'])
            sample['retrieved_policies'] = row['retrieved_policies']
            samples.append(sample)
        
        return {
            "count": len(flagged_df),
            "total_impact": float(flagged_df['Final_Total'].sum()),
            "samples": samples
        }
    except Exception as e:
        logger.error(f"Detect anomalies failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model-anomalies")
async def model_based_anomalies(filter_request: FilterRequest):
    try:
        filtered_df = apply_filters(df, filter_request.filters) if filter_request.filters else df.copy()
        logger.info(f"Filtered data size: {len(filtered_df)} rows")
        flagged_df = run_anomaly_detection(filtered_df)
        
        if filter_request.severity and len(flagged_df) > 0:
            flagged_df = flagged_df[flagged_df['severity'].isin(filter_request.severity)]

        samples = []
        for _, row in flagged_df.sort_values('severity_score', ascending=False).head(1000).iterrows():
            sample = {col: convert_value(val) for col, val in row.items() if col != 'embeddings'}
            samples.append(sample)

        return {
            "count": len(flagged_df),
            "total_impact": float(flagged_df['Final_Total'].sum()),
            "samples": samples,
            "severity_distribution": flagged_df['severity'].value_counts().to_dict()
        }
    except Exception as e:
        logger.error(f"Model anomalies failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/report")
async def generate_detailed_report(filter_request: FilterRequest):
    try:
        filtered_df = apply_filters(df, filter_request.filters) if filter_request.filters else df.copy()
        flagged_df = run_anomaly_detection(filtered_df)
        
        report = {
            "anomaly_categories": {},
            "visualization_data": {
                "anomaly_types": {},
                "severity_distribution": {"high": 0, "medium": 0, "low": 0},
                "source_distribution": {}
            },
            "total_stats": {
                "total_records": len(filtered_df),
                "total_anomalies": len(flagged_df),
                "financial_impact": float(flagged_df['Final_Total'].sum())
            }
        }
        
        policy_map = {
            "Incorrect GST Calculation": "incorrect_gst_calculation",
            "Invalid Service Charge for Online Order": "invalid_service_charge",
            "Tax Applied to Non-Taxable Item": "non_taxable_with_tax",
            "Invalid Payment Method for Online Order": "invalid_payment_method",
            "Local Order with Aggregator Data": "local_with_aggregator",
            "Late Cancellation": "late_cancellation",
            "High-Value Cancellation": "high_value_cancellation",
            "Unapproved Complimentary Order": "unapproved_complimentary",
            "Model Detected Anomaly": "model_detected_anomaly"
        }

        for _, row in flagged_df.iterrows():
            anomalies = (row['detected_flags'] + "|" + row['model_flags']).split('|')
            for anomaly in anomalies:
                if not anomaly.strip():
                    continue
                    
                category = policy_map.get(anomaly.strip())
                if category:
                    if category not in report["anomaly_categories"]:
                        if "Model" in anomaly:
                            policy = {
                                "policy_name": "Machine Learning Detected Anomaly",
                                "description": "Anomaly detected by AI model based on complex patterns",
                                "severity": row['severity'],
                                "action": "Review transaction for unusual patterns"
                            }
                        else:
                            policy = next(
                                (p for p in AUDIT_POLICIES if p['policy_name'].startswith(anomaly.split()[0])),
                                {"policy_name": anomaly, "description": "Unknown anomaly", "action": "Investigate"}
                            )
                        
                        report["anomaly_categories"][category] = {
                            "records": [],
                            "count": 0,
                            "severity": row['severity'],
                            "policy_details": policy
                        }
                    
                    sample = {col: convert_value(val) for col, val in row.items() if col != 'embeddings'}
                    sample['severity_score'] = float(row['severity_score'])
                    report["anomaly_categories"][category]["records"].append(sample)
                    report["anomaly_categories"][category]["count"] += 1

        for category, data in report["anomaly_categories"].items():
            policy_name = data["policy_details"].get("policy_name", "Unknown")
            report["visualization_data"]["anomaly_types"][policy_name] = data["count"]
            report["visualization_data"]["severity_distribution"][data["severity"]] += data["count"]
            source_counts = pd.DataFrame(data["records"])['Order_Source'].value_counts().to_dict()
            for source, count in source_counts.items():
                report["visualization_data"]["source_distribution"][source] = \
                    report["visualization_data"]["source_distribution"].get(source, 0) + count

        for category, data in report["anomaly_categories"].items():
            if data["count"] > 0:
                policy = data["policy_details"]
                if "Model" in policy.get("policy_name", ""):
                    prompt = f"""
                    Analyze these {data['count']} machine learning-detected anomalies in restaurant transactions.
                    Common patterns in these anomalies:
                    {json.dumps(pd.DataFrame(data['records']).describe(), indent=2)}
                    
                    Provide:
                    1. Root cause analysis
                    2. 3 specific recommendations
                    3. Potential business impact
                    """
                else:
                    example = next(
                        (r for r in data["records"] if all(k in r for k in policy.get("relevant_columns", []))),
                        {}
                    )
                    prompt = f"""
                    Generate improvement recommendations for {policy['policy_name']}:
                    - Description: {policy['description']}
                    - Severity: {data['severity']}
                    - Example Case: {json.dumps(example, indent=2)}
                    - Suggested Action: {policy['action']}
                    
                    Provide:
                    1. Root cause analysis
                    2. 3 specific improvement steps
                    3. Expected business impact
                    """
                try:
                    response = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "You are a financial audit expert."},
                            {"role": "user", "content": prompt}
                        ],
                        model="llama3-70b-8192",
                        temperature=0.3,
                        max_tokens=500
                    )
                    data["recommendations"] = response.choices[0].message.content
                except Exception as e:
                    data["recommendations"] = f"Recommendations unavailable due to API error: {str(e)}"
        
        return report
    except Exception as e:
        logger.error(f"Generate detailed report failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_with_llama(request: ChatRequest):
    try:
        logger.info(f"Received message: {request.message}")
        
        flagged_df = run_anomaly_detection(df)
        anomaly_counts = flagged_df['detected_flags'].value_counts().to_dict()
        total_anomalies = len(flagged_df)
        
        context = (
            "You are assisting a manager with fraud and anomaly detection in restaurant transactions.\n"
            "Here’s the current dataset summary:\n"
            f"- Total records analyzed: {len(df)}\n"
            f"- Total anomalies detected: {total_anomalies}\n"
            f"- Common issues: {json.dumps(anomaly_counts, indent=2)}\n"
            "Use this data to provide professional, relevant responses. Stay focused on fraud/anomaly detection."
        )

        prompt = (
            f"{context}\n\n"
            f"User: {request.message}\n"
            "Respond professionally, sticking to the scope of fraud and anomaly detection."
        )

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a professional assistant specializing in fraud and anomaly detection."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192",
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )

        chatbot_response = response.choices[0].message.content
        logger.info(f"Generated response: {chatbot_response}")

        result = {"response": chatbot_response}
        if "chart" in request.message.lower() or "visualization" in request.message.lower():
            labels = list(anomaly_counts.keys())
            data = list(anomaly_counts.values())
            chart_config = {
                "type": "bar",
                "data": {
                    "labels": labels,
                    "datasets": [{"label": "Anomaly Counts", "data": data}]
                }
            }
            chart_str = json.dumps(chart_config).replace(" ", "")
            encoded_chart = urllib.parse.quote(chart_str)
            chart_url = f"https://quickchart.io/chart?c={encoded_chart}"
            result["chart_url"] = chart_url
            logger.info(f"Added chart URL to response: {chart_url}")

        return result

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)
