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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = FastAPI()

logger.info("Starting FastAPI application")

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

@app.post("/users")
def add_users():
    logger.info("POST /users - Starting user data processing")
    
    # Generate random users
    users = [{"id": i, "name": f"User{i}", "age": 20 + i} for i in range(1, 31)]
    logger.info(f"Generated {len(users)} sample users")
    
    df = pd.DataFrame(users)
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
        chunk_path = f"/data/chunks/users_part{idx}.parquet"
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
        "message": "Users added to IPFS", 
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
        
        # Filter and return results
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

@app.get("/users")
def get_users():
    if not chunk_metadata:
        logger.warning("No user data available")
        return {"error": "No user data available."}

    logger.info(f"Starting processing of {len(chunk_metadata)} chunks")
    
    # Process chunks in parallel
    processed_users = []
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
                    processed_users.extend(result['data'])
                    if "source" in result:
                        sources.append(result["source"])
            except Exception as e:
                logger.error(f"Worker {worker_id} raised exception: {str(e)}")
    
    logger.info(f"Retrieved {len(processed_users)} users over 30")
    return {
        "users_over_30": processed_users,
        "sources": sources,
        "total_records": len(processed_users)
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