from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from groq import Groq
import os
import json
from typing import Dict, List, Optional

app = FastAPI()
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
ENV_PATH = os.path.join(ROOT_DIR, ".env")

DEFAULT_THRESHOLDS = {
    "tax_variance": 1,
    "high_severity_tax_variance": 50,
    "cancellation_minutes": 30,
    "high_value_cancellation": 1000,
    "high_value_complimentary": 500,
    "service_charge_online": 0,
    "non_taxable_tax_threshold": 0
}


load_dotenv(ENV_PATH)
client = Groq(api_key= os.getenv("GROQ_API_KEY"))
df = None

# Load audit policies
with open('audit_policies.json') as f:
    AUDIT_POLICIES = json.load(f)['audit_policies']

# Load data on startup
@app.on_event("startup")
async def startup_event():
    global df
    try:
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
 
        df.fillna({
            'Discount': 0,
            'Final_Total': 0,
            'Tax': 0,
            'Sub_Total': 0,
            'CGST_Amount': 0,
            'SGST_Amount': 0,
            'VAT_Amount': 0,
            'Service_Charge_Amount': 0,
            'Online_Tax_Calculated_s': 0,
            'Online_Tax_Calculated_z': 0,
            'Cancelled_Invoice_Total_co': 0,
            'Food_Preparation_Time_Z': 0,
            'Food_Preparation_Time_S': 0
        }, inplace=True)
        
        # Create derived columns
        df['Is_Online'] = df['Order_From_z'].notna() | df['Order_From_s'].notna()
        df['Order_Source'] = np.where(
            df['Order_From_z'].notna(), 'Zomato',
            np.where(df['Order_From_s'].notna(), 'Swiggy', 'Local'))
        
        print("Data loaded successfully")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

class FilterRequest(BaseModel):
    filters: Optional[Dict[str, List[str]]] = None
    severity: Optional[List[str]] = None
    thresholds: Optional[Dict[str, float]] = None



@app.get("/filters/options")
async def get_filter_options():
    """Get available filter values for frontend"""
    return {
        "Order_Type": df['Order_Type'].cat.categories.tolist(),
        "Payment_Type": df['Payment_Type'].cat.categories.tolist(),
        "Order_Source": ["Zomato", "Swiggy", "Local"],
        "Status": df['Status'].cat.categories.tolist(),
        "Severity": ["high", "medium", "low"]
    }





@app.post("/audit-report")
async def generate_audit_report(filter_request: FilterRequest):
    """Generate comprehensive audit report with categorized anomalies"""
    try:
        thresholds = DEFAULT_THRESHOLDS.copy()
        if filter_request.thresholds:
            thresholds.update(filter_request.thresholds)
            
        # Apply filters
        filtered_df = apply_filters(df, filter_request.filters) if filter_request.filters else df.copy()
        
        # Detect anomalies using existing logic
        flagged_df = run_anomaly_detection(filtered_df,thresholds)
        
        samples = []
        for _, row in flagged_df.iterrows():
            sample = {}
            for col, value in row.items():
                if pd.isna(value):
                    sample[col] = None
                elif isinstance(value, (np.floating, float)):
                    sample[col] = float(value)
                elif isinstance(value, (np.integer, int)):
                    sample[col] = int(value)
                else:
                    sample[col] = str(value) if value is not None else None
            samples.append(sample)
            
        flagged_df = pd.DataFrame(samples)

        
        
        # Initialize response structure
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
        
        flagged_df = flagged_df.head(1000)

        # Map existing anomalies to policy categories
        policy_map = {
            "Incorrect GST Calculation": "incorrect_gst_calculation",
            "Invalid Service Charge for Online Order": "invalid_service_charge",
            "Tax Applied to Non-Taxable Item": "non_taxable_with_tax",
            "Invalid Payment Method for Online Order": "invalid_payment_method",
            "Local Order with Aggregator Data": "local_with_aggregator",
            "Late Cancellation": "late_cancellation",
            "High-Value Cancellation": "high_value_cancellation",
            "Unapproved Complimentary Order": "unapproved_complimentary"
        }

        # Group anomalies by policy category
        for _, row in flagged_df.iterrows():
            anomalies = row['anomalies'].split('|')
            for anomaly in anomalies:
                category = policy_map.get(anomaly.strip())
                if category:
                    if category not in report["anomaly_categories"]:
                        report["anomaly_categories"][category] = {
                            "records": [],
                            "count": 0,
                            "severity": row['severity'],
                            "policy_details": next(
                                (p for p in AUDIT_POLICIES if p['policy_name'].startswith(anomaly.split()[0])), 
                                {}
                            )
                        }
                    
                    sample = {}
                    for col, value in row.items():
                        if pd.isna(value):
                            sample[col] = None
                        elif isinstance(value, (np.floating, float)):
                            sample[col] = float(value)
                        elif isinstance(value, (np.integer, int)):
                            sample[col] = int(value)
                        else:
                            sample[col] = str(value) if value is not None else None
                    
                    report["anomaly_categories"][category]["records"].append(sample)
                    report["anomaly_categories"][category]["count"] += 1

        # Generate visualization data
        for category, data in report["anomaly_categories"].items():
            # Anomaly type counts
            policy_name = data["policy_details"].get("policy_name", "Unknown")
            report["visualization_data"]["anomaly_types"][policy_name] = data["count"]
            
            # Severity distribution
            report["visualization_data"]["severity_distribution"][data["severity"]] += data["count"]
            
            # Source distribution
            source_counts = pd.DataFrame(data["records"])['Order_Source'].value_counts().to_dict()
            for source, count in source_counts.items():
                report["visualization_data"]["source_distribution"][source] = \
                    report["visualization_data"]["source_distribution"].get(source, 0) + count

        # # Add LLM recommendations
        for category, data in report["anomaly_categories"].items():
            if data["count"] > 0:
                policy = data["policy_details"]
                example = next(
                    (r for r in data["records"] if all(k in r for k in policy["relevant_columns"])),
                    {}
                )
                
                prompt = f"""
                Generate improvement recommendations for {policy['policy_name']}:
                - Description: {policy['description']}
                - Severity: {data['severity']}
                - Example Case: {json.dumps(example)}
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
                    data["recommendations"] = f"Recommendation generation failed: {str(e)}"
            
        return report

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect-anomalies")
async def detect_anomalies(filter_request: FilterRequest):
    """Detect anomalies with filters"""
    try:
        filtered_df = apply_filters(df, filter_request.filters)
        flagged_df = run_anomaly_detection(filtered_df)
        
        if filter_request.severity:
            flagged_df = flagged_df[flagged_df['severity'].isin(filter_request.severity)]
        
        samples = []
        for _, row in flagged_df.head(1000).iterrows():
            sample = {}
            for col, value in row.items():
                if pd.isna(value):
                    sample[col] = None
                elif isinstance(value, (np.floating, float)):
                    sample[col] = float(value)
                elif isinstance(value, (np.integer, int)):
                    sample[col] = int(value)
                else:
                    sample[col] = str(value) if value is not None else None
            samples.append(sample)
        
        return {
            "count": len(flagged_df),
            "total_impact": float(flagged_df['Final_Total'].sum()),
            "samples": samples
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




def apply_filters(data: pd.DataFrame, filters: Dict[str, List[str]]) -> pd.DataFrame:
    """Apply multiple filters to DataFrame"""
    
    if not filters:
        return data.copy()
    
    mask = pd.Series(True, index=data.index)
    for col, values in filters.items():
        if col in data.columns:
            mask &= data[col].isin(values)
    return data[mask].copy()

def run_anomaly_detection(data: pd.DataFrame, thresholds: Dict[str, float]) -> pd.DataFrame:
    """Enhanced anomaly detection with configurable thresholds"""
    df = data.copy()
    df['anomalies'] = ""
    df['severity'] = "medium"
    
    # 1. Tax Discrepancies
    gst_mask = (
        (df['Order_Source'] == 'Local') & 
        (abs(df['CGST_Amount'] + df['SGST_Amount'] - (0.05 * df['Sub_Total'])) > thresholds["tax_variance"]
    ))
    df.loc[gst_mask, 'anomalies'] += "Incorrect GST Calculation|"
    
    # 2. Service Charge for Online Orders
    service_charge_mask = (
        df['Is_Online'] & 
        (df['Service_Charge_Amount'] > thresholds["service_charge_online"])
    )
    df.loc[service_charge_mask, 'anomalies'] += "Invalid Service Charge for Online Order|"
    
    # 3. Non-taxable items with tax
    non_taxable_mask = (
        (df['Non_Taxable'] == 1) & 
        ((df['CGST_Amount'] + df['SGST_Amount'] + df['VAT_Amount']) > thresholds["non_taxable_tax_threshold"])
    )
    df.loc[non_taxable_mask, 'anomalies'] += "Tax Applied to Non-Taxable Item|"
    
    # 4. Payment Method validation
    payment_mask = (
        df['Is_Online'] & 
        ~df['Payment_Type'].isin(['Online', 'Other [ZOMATO PAY]', 'Other [Paytm]'])
    )
    df.loc[payment_mask, 'anomalies'] += "Invalid Payment Method for Online Order|"
    
    # 5. Local orders with aggregator data
    aggregator_mask = (
        (df['Order_Source'] == 'Local') & 
        (df['Aggregator_Order_No_z'].notna() | df['Aggregator_Order_No_s'].notna())
    )
    df.loc[aggregator_mask, 'anomalies'] += "Local Order with Aggregator Data|"
    
    # 6. Late Cancellation
    prep_time_mask = (
        (df['Status'].str.contains('Cancelled')) & 
        (
            (df['Food_Preparation_Time_Z'] > thresholds["cancellation_minutes"]) | 
            (df['Food_Preparation_Time_S'] > thresholds["cancellation_minutes"])
        )
    )
    df.loc[prep_time_mask, 'anomalies'] += "Late Cancellation|"
    
    # 7. High-Value Cancellation
    high_value_mask = (
        (df['Status'].str.contains('Cancelled')) & 
        (df['Cancelled_Invoice_Total_co'] > thresholds["high_value_cancellation"])
    )
    df.loc[high_value_mask, 'anomalies'] += "High-Value Cancellation|"
    
    # 8. Unapproved Complimentary
    complimentary_mask = (
        (df['Status'] == 'Complimentary') & 
        (df['Assign_To'].isna())
    )
    df.loc[complimentary_mask, 'anomalies'] += "Unapproved Complimentary Order|"
    
    df['anomalies'] = df['anomalies'].str.rstrip('|')
    df['severity'] = df.apply(lambda row: calculate_severity(row, thresholds), axis=1)
    
    return df[df['anomalies'] != ""]

def calculate_severity(row, thresholds: Dict[str, float]):
    """Enhanced severity calculation with thresholds"""
    anomalies = row['anomalies'].split('|')
    high_sev = []
    
    for anomaly in anomalies:
        if anomaly == "Incorrect GST Calculation":
            tax_diff = abs(row['CGST_Amount'] + row['SGST_Amount'] - (0.05 * row['Sub_Total']))
            if tax_diff > thresholds["high_severity_tax_variance"]:
                high_sev.append(anomaly)
        elif anomaly == "Unapproved Complimentary Order":
            if row['Final_Total'] > thresholds["high_value_complimentary"]:
                high_sev.append(anomaly)
        elif anomaly == "High-Value Cancellation":
            high_sev.append(anomaly)
    
    if high_sev:
        return "high"
    
    medium_sev = [
        "Invalid Payment Method for Online Order",
        "Late Cancellation",
        "Tax Applied to Non-Taxable Item"
    ]
    
    if any(anom in anomalies for anom in medium_sev):
        return "medium"
    
    return "low"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8008,reload=True)