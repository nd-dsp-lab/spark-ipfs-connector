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

logger.info("Initializing Spark Session")
spark = SparkSession.builder \
    .appName("FastAPISparkDriver") \
    .master("spark://spark-master:7077") \
    .config("spark.python.worker.reuse", "true") \
    .config("spark.pyspark.python", "/usr/bin/python3") \
    .config("spark.pyspark.driver.python", "/usr/bin/python3") \
    .getOrCreate()
logger.info(f"Spark Session created: {spark.sparkContext.appName}")

user_schema = StructType([
    StructField("id", IntegerType(), False),
    StructField("name", StringType(), False),
    StructField("age", IntegerType(), False)
])
logger.info("User schema defined")

chunk_cids = []

@app.post("/users")
def add_users():
    logger.info("POST /users - Starting user data processing")
    
    # Generate random users
    users = [{"id": i, "name": f"User{i}", "age": 20 + i} for i in range(1, 31)]
    logger.info(f"Generated {len(users)} sample users")
    
    df = pd.DataFrame(users)
    chunk_cids.clear()
    logger.info("Cleared previous chunk CIDs")

    # Split into 3 chunks
    chunks = np.array_split(df, 3)
    logger.info(f"Split data into {len(chunks)} chunks")
    
    # Iterate over the chunks, write to Parquet, and upload to IPFS
    for idx, chunk in enumerate(chunks, start=1):
        chunk_path = f"/data/chunk{idx}/users_part{idx}.parquet"
        os.makedirs(f"/data/chunk{idx}", exist_ok=True)
        logger.info(f"Created directory for chunk {idx}")
        
        logger.info(f"Writing chunk {idx} to Parquet file at {chunk_path}")
        pq.write_table(pa.Table.from_pandas(chunk), chunk_path)

        # Upload chunk to IPFS
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
    if not chunk_cids:
        logger.warning("No user data available")
        return {"error": "No user data available."}

    # Broadcast the chunk CIDs to all workers
    broadcast_cids = spark.sparkContext.broadcast(chunk_cids)
    logger.info("Broadcasted chunk CIDs to workers")

    # UDF to fetch, process data, and return filtered results
    @udf(returnType=StructType([
        StructField("status", StringType(), False),
        StructField("worker_id", IntegerType(), False),
        StructField("data", StringType(), True)
    ]))
    def process_ipfs(worker_id):
        """Worker processes assigned chunk from broadcasted CIDs and returns data"""
        import requests, os, tempfile
        import pandas as pd
        import logging
       
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(f"worker-{worker_id}")
       
        logger.info(f"Worker {worker_id} starting processing")
        try:
            # Get CID from broadcasted list
            cid_list = broadcast_cids.value
            chunk_cid = cid_list[worker_id % len(cid_list)]
            urls = {f"node{i}": f"http://ipfs{i}:8080/ipfs/{chunk_cid}" for i in range(1, 4)}
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

    # Create worker DataFrame
    logger.info("Creating worker DataFrame")
    worker_df = spark.createDataFrame([(i,) for i in range(3)], ["worker_id"])
   
    # Process using UDF
    result_df = worker_df.withColumn("result", process_ipfs(col("worker_id")))
    results = result_df.collect()
   
    processed_users = []
    for row in results:
        print(row.result)
        status, worker_id, data_json = row.result
        if status == "SUCCESS" and data_json:
            try:
                users = json.loads(data_json)
                processed_users.extend(users)
            except json.JSONDecodeError:
                logger.error(f"Worker {worker_id} returned invalid JSON")
   
    logger.info(f"Retrieved {len(processed_users)} users over 30")
    return {"users_over_30": processed_users}