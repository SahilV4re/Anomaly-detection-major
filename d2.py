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
import uvicorn

client = Groq(api_key="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")
df = None
model_pipeline = None

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

with open('audit_policies.json') as f:
    AUDIT_POLICIES = json.load(f)['audit_policies']

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
    global df, model_pipeline
    try:
        df = pd.read_csv(
            "Hackathon Dataset.csv",  # Adjust path
            parse_dates=['Date', 'Timestamp', 'Cancelled_Time_z', 'Cancelled_Time_s'],
            dtype={
                'Final_Total': 'float32',
                'Discount': 'float32',
                'Order_Type': 'category',
                'Payment_Type': 'category',
                'Status': 'category'
            }
        )
        # df = df[:1000]
        df.fillna({
            'Discount': 0, 'Final_Total': 0, 'Tax': 0, 'Sub_Total': 0,
            'CGST_Amount': 0, 'SGST_Amount': 0, 'VAT_Amount': 0,
            'Service_Charge_Amount': 0, 'Online_Tax_Calculated_s': 0,
            'Online_Tax_Calculated_z': 0, 'Cancelled_Invoice_Total_co': 0,
            'Food_Preparation_Time_Z': 0, 'Food_Preparation_Time_S': 0
        }, inplace=True)

        df['Is_Online'] = (df['Order_From_z'].notna() | df['Order_From_s'].notna()).astype(int)
        df['Order_Source'] = np.where(
            df['Order_From_z'].notna(), 'Zomato',
            np.where(df['Order_From_s'].notna(), 'Swiggy', 'Local'))

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
                n_estimators=200
            ))
        ])

        model_pipeline.fit(df)
        print("✅ Data loaded and model trained successfully")
    except Exception as e:
        print(f"❌ Error during startup: {str(e)}")
        raise
    yield

app = FastAPI(lifespan=lifespan)

class FilterRequest(BaseModel):
    filters: Optional[Dict[str, List[str]]] = None
    severity: Optional[List[str]] = None

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

def compute_severity_score(row):
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

    total_score = (w_financial * financial_score + 
                   w_operational * operational_score + 
                   w_statistical * statistical_score + rule_boost)
    return min(max(total_score * 100, 0), 100)

def run_anomaly_detection(data: pd.DataFrame) -> pd.DataFrame:
    df_local = data.copy()
    df_local['detected_flags'] = ""
    df_local['model_flags'] = ""
    df_local['severity'] = "medium"
    df_local['severity_score'] = 0.0

    if 'Non_Taxable' in df_local.columns:
        non_taxable_mask = (
            (df_local['Non_Taxable'] == 1) & 
            ((df_local['CGST_Amount'] + df_local['SGST_Amount'] + df_local['VAT_Amount']) > 0)
        )
        df_local.loc[non_taxable_mask, 'detected_flags'] += "Tax Applied to Non-Taxable Item|"

    if 'Aggregator_Order_No_z' in df_local.columns and 'Aggregator_Order_No_s' in df_local.columns:
        aggregator_mask = (
            (df_local['Order_Source'] == 'Local') & 
            (df_local['Aggregator_Order_No_z'].notna() | df_local['Aggregator_Order_No_s'].notna())
        )
        df_local.loc[aggregator_mask, 'detected_flags'] += "Local Order with Aggregator Data|"

    gst_mask = (
        (df_local['Order_Source'] == 'Local') & 
        (np.abs(df_local['CGST_Amount'] + df_local['SGST_Amount'] - (0.05 * df_local['Sub_Total'])) > 1)
    )
    df_local.loc[gst_mask, 'detected_flags'] += "Incorrect GST Calculation|"

    service_charge_mask = (
        df_local['Is_Online'].astype(bool) & 
        (df_local['Service_Charge_Amount'] > 0)
    )
    df_local.loc[service_charge_mask, 'detected_flags'] += "Invalid Service Charge for Online Order|"

    payment_mask = (
        df_local['Is_Online'].astype(bool) & 
        ~df_local['Payment_Type'].isin(['Online', 'Other [ZOMATO PAY]', 'Other [Paytm]'])
    )
    df_local.loc[payment_mask, 'detected_flags'] += "Invalid Payment Method for Online Order|"

    prep_time_mask = (
        (df_local['Status'].str.contains('Cancelled', na=False)) & 
        ((df_local['Food_Preparation_Time_Z'] > 30) | (df_local['Food_Preparation_Time_S'] > 30))
    )
    df_local.loc[prep_time_mask, 'detected_flags'] += "Late Cancellation|"

    high_value_mask = (
        (df_local['Status'].str.contains('Cancelled', na=False)) & 
        (df_local['Cancelled_Invoice_Total_co'] > 1000)
    )
    df_local.loc[high_value_mask, 'detected_flags'] += "High-Value Cancellation|"

    if 'Assign_To' in df_local.columns:
        complimentary_mask = (
            (df_local['Status'] == 'Complimentary') & 
            (df_local['Assign_To'].isna())
        )
        df_local.loc[complimentary_mask, 'detected_flags'] += "Unapproved Complimentary Order|"

    if model_pipeline is not None:
        try:
            if data[numerical_features].dropna().empty:
                print("No valid numerical data for model")
            else:
                scores = model_pipeline.decision_function(df_local)
                predictions = model_pipeline.predict(df_local)
                df_local['model_anomaly_score'] = scores
                model_anomalies = predictions == -1
                df_local.loc[model_anomalies, 'model_flags'] += "Model Detected Anomaly|"
        except Exception as e:
            print(f"Model detection failed: {str(e)}")

    df_local['detected_flags'] = df_local['detected_flags'].str.rstrip('|')
    df_local['model_flags'] = df_local['model_flags'].str.rstrip('|')

    flagged_df = df_local[(df_local['detected_flags'] != "") | (df_local['model_flags'] != "")]
    if len(flagged_df) > 0:
        flagged_df['severity_score'] = flagged_df.apply(compute_severity_score, axis=1)
        flagged_df['severity'] = pd.cut(
            flagged_df['severity_score'],
            bins=[0, 33, 66, 100],
            labels=['low', 'medium', 'high'],
            include_lowest=True
        )
        df_local.loc[flagged_df.index, 'severity_score'] = flagged_df['severity_score']
        df_local.loc[flagged_df.index, 'severity'] = flagged_df['severity']

    return flagged_df

@app.post("/audit-report")
async def generate_audit_report(filter_request: FilterRequest):
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
                                {}
                            )
                        
                        report["anomaly_categories"][category] = {
                            "records": [],
                            "count": 0,
                            "severity": row['severity'],
                            "policy_details": policy
                        }
                    
                    sample = {col: convert_value(val) for col, val in row.items()}
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
                    data["recommendations"] = "Recommendations unavailable due to API error"
        
        return report
    except Exception as e:
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
            sample = {col: convert_value(val) for col, val in row.items()}
            sample['severity_score'] = float(row['severity_score'])
            samples.append(sample)
        
        return {
            "count": len(flagged_df),
            "total_impact": float(flagged_df['Final_Total'].sum()),
            "samples": samples
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model-anomalies")
async def model_based_anomalies(filter_request: FilterRequest):
    try:
        filtered_df = apply_filters(df, filter_request.filters) if filter_request.filters else df.copy()
        print(f"Filtered data size: {len(filtered_df)} rows")

        required_cols = numerical_features + categorical_features
        missing_cols = [col for col in required_cols if col not in filtered_df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in filtered data: {missing_cols}")

        local_pipeline = Pipeline([
            ('preprocessor', model_pipeline.named_steps['preprocessor']),
            ('detector', IsolationForest(
                contamination=0.2,
                random_state=42,
                n_estimators=200
            ))
        ])
        local_pipeline.fit(filtered_df)

        scores = local_pipeline.decision_function(filtered_df)
        predictions = local_pipeline.predict(filtered_df)
        print(f"Total anomalies predicted: {(predictions == -1).sum()}")

        df_copy = filtered_df.copy()
        df_copy['model_anomaly_score'] = scores
        df_copy['is_anomaly'] = predictions == -1
        flagged_df = df_copy[df_copy['is_anomaly']]

        if len(flagged_df) > 0:
            flagged_df['severity_score'] = flagged_df.apply(compute_severity_score, axis=1)
            flagged_df['severity'] = pd.cut(
                flagged_df['severity_score'],
                bins=[0, 33, 66, 100],
                labels=['low', 'medium', 'high'],
                include_lowest=True
            )
            print(f"Severity score range: {flagged_df['severity_score'].min()} to {flagged_df['severity_score'].max()}")

        if filter_request.severity and len(flagged_df) > 0:
            flagged_df = flagged_df[flagged_df['severity'].isin(filter_request.severity)]

        samples = []
        for _, row in flagged_df.sort_values('severity_score', ascending=False).head(1000).iterrows():
            samples.append({col: convert_value(val) for col, val in row.items()})

        return {
            "count": len(flagged_df),
            "total_impact": float(flagged_df['Final_Total'].sum()),
            "samples": samples,
            "severity_distribution": flagged_df['severity'].value_counts().to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8008, reload=True)