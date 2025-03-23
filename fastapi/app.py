from fastapi import FastAPI, HTTPException
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import json
import os
import numpy as np
import logging
import concurrent.futures
import tempfile
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = FastAPI()

logger.info("Starting FastAPI application for patient records")

# Environment variables with defaults that prioritize local access
LOCAL_IPFS_API = os.environ.get("LOCAL_IPFS_API", "http://ipfs:5001")
LOCAL_IPFS_GATEWAY = os.environ.get("LOCAL_IPFS_GATEWAY", "http://ipfs:8080")
USE_PUBLIC_GATEWAYS = os.environ.get("USE_PUBLIC_GATEWAYS", "true").lower() == "true"

# Public gateways as fallback (will only be used if local gateway fails)
PUBLIC_GATEWAYS = [
    "https://ipfs.io/ipfs/",
    "https://dweb.link/ipfs/",
    "https://gateway.ipfs.io/ipfs/",
    "https://cloudflare-ipfs.com/ipfs/"
]

# Store metadata about uploaded chunks
chunk_metadata = []

# Sample data for demonstration
disease_types = ["Hypertension", "Diabetes", "Asthma", "Arthritis", "COPD", "Migraine", "Influenza"]
hospital_ids = ["H001", "H002", "H003", "H004", "H005"]
doctor_ids = ["D001", "D002", "D003", "D004", "D005", "D006", "D007", "D008", "D009", "D010"]

def generate_sample_date(start_date, end_date=None):
    """Generate a sample date between start_date and end_date (or today)"""
    if end_date is None:
        end_date = datetime.now()
    
    time_between_dates = end_date - start_date
    days_between = time_between_dates.days
    random_days = random.randrange(days_between)
    return start_date + timedelta(days=random_days)

def generate_diagnosis_report(disease_type):
    """Generate a simple diagnosis report based on disease type"""
    reports = {
        "Hypertension": "Patient shows elevated blood pressure readings. Recommend lifestyle changes and monitoring.",
        "Diabetes": "Blood glucose levels indicate Type 2 diabetes. Dietary changes and medication advised.",
        "Asthma": "Pulmonary function tests confirm asthma diagnosis. Inhaler prescribed for management.",
        "Arthritis": "X-rays show joint inflammation consistent with arthritis. Anti-inflammatory medication recommended.",
        "COPD": "Decreased lung function detected. Long-term management plan established.",
        "Migraine": "Recurring severe headaches with associated symptoms. Trigger identification important.",
        "Influenza": "Positive for influenza virus. Symptomatic treatment and rest advised."
    }
    return reports.get(disease_type, "Diagnosis pending additional tests.")

def generate_prescription(disease_type):
    """Generate a sample prescription based on disease type"""
    prescriptions = {
        "Hypertension": "Lisinopril 10mg daily, monitor blood pressure weekly",
        "Diabetes": "Metformin 500mg twice daily with meals, check blood sugar regularly",
        "Asthma": "Albuterol inhaler as needed, Fluticasone 220mcg twice daily",
        "Arthritis": "Ibuprofen 400mg three times daily with food, physical therapy",
        "COPD": "Tiotropium inhaler daily, pulmonary rehabilitation program",
        "Migraine": "Sumatriptan 50mg as needed for migraine onset, max 100mg daily",
        "Influenza": "Oseltamivir 75mg twice daily for 5 days, increased fluid intake"
    }
    return prescriptions.get(disease_type, "Prescription pending diagnosis.")

@app.post("/patients")
def add_patients():
    logger.info("POST /patients - Starting patient data processing")
    
    # Generate random patient records
    patients = []
    for i in range(1, 31):
        # Generate admission date within last 2 years
        two_years_ago = datetime.now() - timedelta(days=2*365)
        admission_date = generate_sample_date(two_years_ago)
        
        # Generate release date after admission date (or None if still admitted)
        release_date = None
        if random.random() > 0.2:  # 80% of patients have been released
            max_stay = min(60, (datetime.now() - admission_date).days)  # Max 60 days stay
            if max_stay > 0:
                stay_days = random.randint(1, max_stay)
                release_date = admission_date + timedelta(days=stay_days)
        
        # Select a random disease type
        disease_type = random.choice(disease_types)
        
        # Create patient record
        patient = {
            "patient_id": f"P{i:03d}",
            "doctor_id": random.choice(doctor_ids),
            "hospital_id": random.choice(hospital_ids),
            "age": random.randint(18, 90),
            "admission_date": admission_date.strftime("%Y-%m-%d"),
            "release_date": release_date.strftime("%Y-%m-%d") if release_date else None,
            "disease_type": disease_type,
            "diagnosis_report": generate_diagnosis_report(disease_type),
            "prescription": generate_prescription(disease_type)
        }
        patients.append(patient)
    
    logger.info(f"Generated {len(patients)} sample patient records")
    
    df = pd.DataFrame(patients)
    chunk_metadata.clear()
    logger.info("Cleared previous chunk metadata")

    # Split into 3 chunks
    chunks = np.array_split(df, 3)
    logger.info(f"Split data into {len(chunks)} chunks")
    
    # Directory for chunks
    os.makedirs("/data/chunks", exist_ok=True)
    
    # Iterate over the chunks, write to Parquet, and upload to IPFS
    for idx, chunk in enumerate(chunks, start=1):
        # Save locally first - we'll use this as fallback
        chunk_path = f"/data/chunks/patients_part{idx}.parquet"
        logger.info(f"Writing chunk {idx} to Parquet file at {chunk_path}")
        pq.write_table(pa.Table.from_pandas(chunk), chunk_path)

        # Upload chunk to IPFS
        ipfs_api_url = f"{LOCAL_IPFS_API}/api/v0/add"
        logger.info(f"Uploading chunk {idx} to IPFS using {ipfs_api_url}")
        
        try:
            with open(chunk_path, "rb") as f:
                response = requests.post(ipfs_api_url, files={"file": f})
                response.raise_for_status()
                cid = response.json()["Hash"]
                
                # Store metadata including local path for fallback
                chunk_metadata.append({
                    "cid": cid,
                    "local_path": chunk_path,
                    "index": idx
                })
                
                logger.info(f"Chunk {idx} uploaded to IPFS with CID: {cid}")
                
                # Pin content to ensure it stays in the local node
                pin_url = f"{LOCAL_IPFS_API}/api/v0/pin/add?arg={cid}"
                requests.post(pin_url)
                logger.info(f"Pinned CID {cid} to local node")
                
        except Exception as e:
            logger.error(f"Error uploading chunk {idx} to IPFS: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to upload chunk {idx}: {str(e)}")

    logger.info(f"All chunks uploaded to IPFS. Total chunks: {len(chunk_metadata)}")
    return {
        "message": "Patient records added to IPFS", 
        "cids": [item["cid"] for item in chunk_metadata]
    }

def read_local_file(file_path: str) -> Optional[pd.DataFrame]:
    """Attempt to read a local Parquet file"""
    try:
        logger.info(f"Attempting to read local file at {file_path}")
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            logger.info(f"Successfully read local file: {file_path}")
            return df
        return None
    except Exception as e:
        logger.warning(f"Failed to read local file {file_path}: {str(e)}")
        return None

def fetch_from_ipfs(cid: str, timeout: int = 5) -> Optional[bytes]:
    """Try to fetch content from IPFS with multiple fallback options"""
    # First try local gateway - this should be fast
    local_url = f"{LOCAL_IPFS_GATEWAY}/ipfs/{cid}"
    
    try:
        logger.info(f"Fetching from local IPFS gateway: {local_url}")
        response = requests.get(local_url, timeout=timeout)
        if response.status_code == 200:
            logger.info(f"Successfully fetched {cid} from local gateway")
            return response.content
    except Exception as e:
        logger.warning(f"Failed to fetch from local gateway: {str(e)}")
    
    # If local fails and public gateways are enabled, try them
    if USE_PUBLIC_GATEWAYS:
        for gateway in PUBLIC_GATEWAYS:
            try:
                url = f"{gateway}{cid}"
                logger.info(f"Fetching from public gateway: {url}")
                response = requests.get(url, timeout=timeout)
                if response.status_code == 200:
                    logger.info(f"Successfully fetched {cid} from {gateway}")
                    return response.content
            except Exception as e:
                logger.warning(f"Failed to fetch from {gateway}: {str(e)}")
            # Small delay before trying next gateway
            time.sleep(0.5)
    
    return None

def process_chunk(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Process a chunk using multiple fallback methods"""
    cid = metadata["cid"]
    local_path = metadata["local_path"]
    worker_id = metadata["index"]
    
    logger.info(f"Worker {worker_id} processing chunk with CID: {cid}")
    
    # Try to get data from IPFS (local gateway first, then public)
    try:
        content = fetch_from_ipfs(cid)
        if content:
            # Save to temp file and process
            output_path = os.path.join(tempfile.gettempdir(), f"{cid}.parquet")
            with open(output_path, "wb") as f:
                f.write(content)
            df = pd.read_parquet(output_path)
            logger.info(f"Worker {worker_id} successfully read data from IPFS")
        else:
            # Fallback to local file if IPFS retrieval failed
            logger.info(f"Worker {worker_id} falling back to local file: {local_path}")
            df = read_local_file(local_path)
            if df is None:
                return {
                    "status": "ERROR", 
                    "worker_id": worker_id, 
                    "message": "Failed to retrieve data from IPFS and local fallback",
                    "data": None
                }
        
        # Filter and return results - keeping the same age filtering as before
        filtered = df[df['age'] > 30]
        data_json = filtered.to_json(orient='records')
        return {
            "status": "SUCCESS", 
            "worker_id": worker_id, 
            "data": json.loads(data_json),
            "source": "IPFS" if content else "Local File"
        }
        
    except Exception as e:
        logger.error(f"Worker {worker_id} encountered error: {str(e)}")
        return {
            "status": "ERROR", 
            "worker_id": worker_id, 
            "message": str(e),
            "data": None
        }

@app.get("/patients")
def get_patients():
    if not chunk_metadata:
        logger.warning("No patient data available")
        return {"error": "No patient data available."}

    logger.info(f"Starting processing of {len(chunk_metadata)} chunks")
    
    # Process chunks in parallel
    processed_patients = []
    sources = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(chunk_metadata)) as executor:
        # Submit processing tasks for each chunk
        future_to_chunk = {executor.submit(process_chunk, metadata): metadata["index"] 
                           for metadata in chunk_metadata}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_chunk):
            worker_id = future_to_chunk[future]
            try:
                result = future.result()
                logger.info(f"Worker {worker_id} completed with status: {result['status']}")
                
                if result['status'] == "SUCCESS" and result['data']:
                    processed_patients.extend(result['data'])
                    if "source" in result:
                        sources.append(result["source"])
            except Exception as e:
                logger.error(f"Worker {worker_id} raised exception: {str(e)}")
    
    logger.info(f"Retrieved {len(processed_patients)} patients over 30")
    return {
        "patients_over_30": processed_patients,
        "sources": sources,
        "total_records": len(processed_patients)
    }

@app.get("/ping")
def ping_ipfs():
    """Check connectivity to local IPFS node"""
    try:
        response = requests.post(f"{LOCAL_IPFS_API}/api/v0/id", timeout=2)
        if response.status_code == 200:
            ipfs_id = response.json()
            return {
                "status": "connected",
                "ipfs_id": ipfs_id.get("ID", "unknown"),
                "addresses": ipfs_id.get("Addresses", [])[:3]  # Show first 3 addresses only
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
    return {"status": "unknown"}