from fastapi import FastAPI
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import json
import os
import numpy as np
from pyspark.sql.functions import col, udf
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = FastAPI()

logger.info("Starting FastAPI application")

# Initialize Spark Session as a standalone cluster client
logger.info("Initializing Spark Session")
spark = SparkSession.builder \
    .appName("FastAPISparkDriver") \
    .master("spark://spark-master:7077") \
    .config("spark.python.worker.reuse", "true") \
    .config("spark.pyspark.python", "/usr/bin/python3") \
    .config("spark.pyspark.driver.python", "/usr/bin/python3") \
    .getOrCreate()
logger.info(f"Spark Session created: {spark.sparkContext.appName}")

# Define user schema
user_schema = StructType([
    StructField("id", IntegerType(), False),
    StructField("name", StringType(), False),
    StructField("age", IntegerType(), False)
])
logger.info("User schema defined")

# Store metadata CID globally
chunk_cids = []
metadata_cid = None

@app.post("/users")
def add_users():
    """Splits users into chunks, saves them as Parquet files, and uploads to IPFS"""
    logger.info("POST /users - Starting user data processing")
    
    users = [{"id": i, "name": f"User{i}", "age": 20 + i} for i in range(1, 31)]
    logger.info(f"Generated {len(users)} sample users")
    
    df = pd.DataFrame(users)
    chunk_cids.clear()
    logger.info("Cleared previous chunk CIDs")

    # Split into 3 chunks
    chunks = np.array_split(df, 3)
    logger.info(f"Split data into {len(chunks)} chunks")
    
    for idx, chunk in enumerate(chunks, start=1):
        chunk_path = f"/data/chunk{idx}/users_part{idx}.parquet"
        os.makedirs(f"/data/chunk{idx}", exist_ok=True)
        logger.info(f"Created directory for chunk {idx}")
        
        logger.info(f"Writing chunk {idx} to Parquet file at {chunk_path}")
        pq.write_table(pa.Table.from_pandas(chunk), chunk_path)

        # Upload Parquet file to IPFS - using container name
        ipfs_api_url = f"http://ipfs{idx}:5001/api/v0/add"
        logger.info(f"Uploading chunk {idx} to IPFS node at {ipfs_api_url}")
        
        try:
            with open(chunk_path, "rb") as f:
                response = requests.post(ipfs_api_url, files={"file": f})
                response.raise_for_status()
                cid = response.json()["Hash"]
                chunk_cids.append(cid)
                logger.info(f"Chunk {idx} uploaded to IPFS with CID: {cid}")
        except Exception as e:
            logger.error(f"Error uploading chunk {idx} to IPFS: {str(e)}")
            raise

    logger.info(f"All chunks uploaded to IPFS. Total CIDs: {len(chunk_cids)}")
    return {"message": "Users added to IPFS", "cids": chunk_cids}

@app.get("/users")
def get_users():
    """Retrieves metadata, distributes work to Spark workers, fetches data, and filters users aged over 30"""
    logger.info("GET /users - Starting user data retrieval and processing")
    
    global metadata_cid
    if not chunk_cids:
        logger.warning("No user data available - chunk_cids is empty")
        return {"error": "No user data available."}

    # Prepare metadata - using container names for URLs
    logger.info("Preparing metadata for IPFS chunks")
    metadata_list = []
    for idx, cid in enumerate(chunk_cids):
        node_id = (idx % 3) + 1  # Round-robin distribution
        metadata_entry = {
            "cid": cid,
            "node_id": node_id,
            "urls": {f"node{i}": f"http://ipfs{i}:8080/ipfs/{cid}" for i in range(1, 4)}
        }
        metadata_list.append(metadata_entry)
        logger.info(f"Added metadata for CID {cid}, assigned to node {node_id}")

    # Upload metadata to IPFS - use ipfs1 container name
    metadata_path = "/data/metadata.json"
    logger.info(f"Writing metadata to {metadata_path}")
    with open(metadata_path, "w") as f:
        json.dump(metadata_list, f)

    ipfs_api_url = "http://ipfs1:5001/api/v0/add"
    logger.info(f"Uploading metadata to IPFS at {ipfs_api_url}")
    try:
        with open(metadata_path, "rb") as f:
            response = requests.post(ipfs_api_url, files={"file": f})
            response.raise_for_status()
            metadata_cid = response.json()["Hash"]
            logger.info(f"Metadata uploaded to IPFS with CID: {metadata_cid}")
    except Exception as e:
        logger.error(f"Error uploading metadata to IPFS: {str(e)}")
        raise

    # UDF to fetch, process data, and return filtered results
    @udf(returnType=StructType([
        StructField("status", StringType(), False),
        StructField("worker_id", IntegerType(), False),
        StructField("data", StringType(), True)
    ]))
    def process_ipfs(worker_id):
        """Worker fetches assigned chunk from IPFS, processes, and returns data"""
        import requests, json, os, tempfile
        import pandas as pd
        import logging
        
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(f"worker-{worker_id}")
        
        logger.info(f"Worker {worker_id} starting processing")
        try:
            # Fetch metadata
            metadata_url = f"http://ipfs1:8080/ipfs/{metadata_cid}"
            logger.info(f"Worker {worker_id} fetching metadata from {metadata_url}")
            metadata_response = requests.get(metadata_url, timeout=10)
            if metadata_response.status_code != 200:
                return ("ERROR: Metadata fetch failed", worker_id, None)
            metadata_list = json.loads(metadata_response.text)
            
            # Assign chunk based on worker_id
            chunk_metadata = metadata_list[worker_id % len(metadata_list)]
            chunk_cid = chunk_metadata["cid"]
            urls = chunk_metadata["urls"]
            logger.info(f"Worker {worker_id} assigned to chunk {chunk_cid}")

            # Fetch chunk data from IPFS
            output_path = os.path.join(tempfile.gettempdir(), f"{chunk_cid}.parquet")
            for node_url in urls.values():
                try:
                    response = requests.get(node_url, timeout=10)
                    if response.status_code == 200:
                        with open(output_path, "wb") as f:
                            f.write(response.content)
                        # Read and process data
                        df = pd.read_parquet(output_path)
                        filtered = df[df['age'] > 30]
                        data_json = filtered.to_json(orient='records')
                        return ("SUCCESS", worker_id, data_json)
                except Exception as e:
                    logger.warning(f"Worker {worker_id} failed node {node_url}: {str(e)}")
            return ("ERROR: All nodes failed", worker_id, None)
        except Exception as e:
            return (f"ERROR: {str(e)}", worker_id, None)

    # Create worker DataFrame and apply UDF
    logger.info("Creating worker DataFrame")
    worker_df = spark.createDataFrame([(i,) for i in range(3)], ["worker_id"])
    result_df = worker_df.withColumn("result", process_ipfs(col("worker_id")))
    results = result_df.collect()
    
    processed_users = []
    for row in results:
        status, worker_id, data_json = row.result
        if status == "SUCCESS" and data_json:
            try:
                users = json.loads(data_json)
                processed_users.extend(users)
            except json.JSONDecodeError:
                logger.error(f"Worker {worker_id} returned invalid JSON")
    
    logger.info(f"Retrieved {len(processed_users)} users over 30")
    return {"users_over_30": processed_users, "metadata_cid": metadata_cid}