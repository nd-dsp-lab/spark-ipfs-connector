# # app.py - FastAPI Spark Driver
# from fastapi import FastAPI
# from pyspark.sql import SparkSession
# import pandas as pd
# import pyarrow as pa
# import pyarrow.parquet as pq

# import requests
# import numpy as np


# app = FastAPI()

# # Initialize Spark Session as a standalone cluster client (Spark master at spark://spark-master:7077)
# spark = spark = SparkSession.builder \
#     .appName("FastAPISparkDriver") \
#     .master("spark://spark-master:7077") \
#     .getOrCreate()


# # Shared list to hold IPFS CIDs for each data chunk
# chunk_cids = []  # [cid1, cid2, cid3]

# @app.post("/users")
# def add_users():
#     # Generate 30 sample user records
#     users = [{"id": i, "name": f"User{i}", "age": 20 + i} for i in range(1, 31)]
#     df = pd.DataFrame(users)
#     chunk_cids.clear()

#     # Split into 3 equal chunks and process each
#     chunks = np.array_split(df, 3)  # 3 chunks of 10 rows each
#     for idx, chunk in enumerate(chunks, start=1):
#         # Save chunk to Parquet file
#         chunk_path = f"/data/chunk{idx}/users_part{idx}.parquet"
#         table = pa.Table.from_pandas(chunk)   # Fixed: using pa instead of pq for Table
#         pq.write_table(table, chunk_path)     # write Parquet file

#         # Add Parquet file to corresponding IPFS node via HTTP API
#         ipfs_api_url = f"http://ipfs{idx}:5001/api/v0/add"
#         with open(chunk_path, "rb") as f:
#             files = {"file": f}
#             response = requests.post(ipfs_api_url, files=files)
#             response.raise_for_status()
#             cid = response.json()["Hash"]      # Content ID from IPFS
#             chunk_cids.append(cid)

#     return {"message": "30 users added and distributed to IPFS nodes", "cids": chunk_cids}

# @app.get("/users")
# def get_users():
#     if not chunk_cids:
#         return {"error": "No user data available. Please add users first."}

#     # Read the three Parquet chunk files as a single Spark DataFrame
#     df = spark.read.parquet("/data/chunk1/users_part1.parquet",
#                              "/data/chunk2/users_part2.parquet",
#                              "/data/chunk3/users_part3.parquet")
#     df.createOrReplaceTempView("users")
#     # Execute the Spark SQL query in parallel across partitions
#     # Inject worker ID
#     result_df = spark.sql("""
#         SELECT *, spark_partition_id() as worker_id 
#         FROM users
#         WHERE age > 30
#     """)
#     # Collect results to driver and convert to list of dicts for JSON output
#     results = [row.asDict() for row in result_df.collect()]
#     return {"users_over_30": results}

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
import tempfile
from pyspark.sql.functions import col, udf, expr

app = FastAPI()

# Initialize Spark Session as a standalone cluster client
spark = SparkSession.builder \
    .appName("FastAPISparkDriver") \
    .master("spark://spark-master:7077") \
    .config("spark.python.worker.reuse", "true") \
    .config("spark.pyspark.python", "/usr/bin/python3") \
    .config("spark.pyspark.driver.python", "/usr/bin/python3") \
    .getOrCreate()

# Define user schema
user_schema = StructType([
    StructField("id", IntegerType(), False),
    StructField("name", StringType(), False),
    StructField("age", IntegerType(), False)
])

# Store metadata CID globally
chunk_cids = []
metadata_cid = None

@app.post("/users")
def add_users():
    """Generates users, partitions data, uploads to IPFS, and stores CIDs"""
    
    users = [{"id": i, "name": f"User{i}", "age": 20 + i} for i in range(1, 31)]
    df = pd.DataFrame(users)
    chunk_cids.clear()

    # Split into 3 chunks
    chunks = np.array_split(df, 3)
    for idx, chunk in enumerate(chunks, start=1):
        chunk_path = f"/data/chunk{idx}/users_part{idx}.parquet"
        os.makedirs(f"/data/chunk{idx}", exist_ok=True)
        pq.write_table(pa.Table.from_pandas(chunk), chunk_path)

        # Upload Parquet file to IPFS
        ipfs_api_url = f"http://ipfs{idx}:5001/api/v0/add"
        with open(chunk_path, "rb") as f:
            response = requests.post(ipfs_api_url, files={"file": f})
            response.raise_for_status()
            cid = response.json()["Hash"]
            chunk_cids.append(cid)

    return {"message": "Users added to IPFS", "cids": chunk_cids}

@app.get("/users")
def get_users():
    """Retrieves metadata, distributes work to Spark workers, fetches data, and filters users aged over 30"""
    
    global metadata_cid
    if not chunk_cids:
        return {"error": "No user data available."}

    # Prepare metadata
    metadata_list = []
    for idx, cid in enumerate(chunk_cids):
        node_id = (idx % 3) + 1  # Round-robin distribution
        metadata_list.append({
            "cid": cid,
            "node_id": node_id,
            "urls": {f"node{i}": f"http://ipfs{i}:8080/ipfs/{cid}" for i in range(1, 4)}
        })

    # Upload metadata to IPFS
    metadata_path = "/data/metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata_list, f)

    ipfs_api_url = "http://ipfs1:5001/api/v0/add"
    with open(metadata_path, "rb") as f:
        response = requests.post(ipfs_api_url, files={"file": f})
        response.raise_for_status()
        metadata_cid = response.json()["Hash"]

    # UDF to fetch & process data from IPFS
    @udf(returnType=StructType([
        StructField("status", StringType(), False),
        StructField("file_path", StringType(), True),
        StructField("chunk_index", IntegerType(), True),
        StructField("worker_id", IntegerType(), True)
    ]))
    def process_ipfs(worker_id):
        """Worker fetches assigned chunk from IPFS and saves locally"""
        
        import requests, json, os, time

        # Fetch metadata
        metadata_list = None
        for attempt in range(3):  # Retry up to 3 times
            try:
                metadata_url = f"http://ipfs1:8080/ipfs/{metadata_cid}"
                metadata_response = requests.get(metadata_url, timeout=10)
                if metadata_response.status_code == 200:
                    metadata_list = json.loads(metadata_response.text)
                    break
                time.sleep(2 ** attempt)  # Exponential backoff
            except:
                continue
        
        if not metadata_list:
            return ("ERROR: Failed to fetch metadata", None, None, worker_id)

        # Assign chunk to worker
        chunk_metadata = metadata_list[int(worker_id) % len(metadata_list)]
        chunk_cid, node_id, urls = chunk_metadata["cid"], chunk_metadata["node_id"], chunk_metadata["urls"]
        output_path = os.path.join(tempfile.gettempdir(), f"{chunk_cid}.parquet")

        # Fetch from IPFS
        for i in range(1, 4):
            try:
                url = urls[f"node{i}"]
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    with open(output_path, "wb") as f:
                        f.write(response.content)
                    return ("SUCCESS", output_path, int(worker_id), worker_id)
            except:
                continue

        return (f"ERROR: Failed to fetch CID {chunk_cid}", None, None, worker_id)

    # Step 4: Create a dummy DataFrame with workers
    worker_df = spark.createDataFrame([(i,) for i in range(3)], ["worker_id"])

    # Step 5: Apply UDF & collect results
    result_df = worker_df.withColumn("result", process_ipfs(col("worker_id")))
    results = result_df.collect()
    processed_results = []

    # Step 6: Process retrieved data
    for row in results:
        status, file_path, chunk_idx, worker_id = row.result
        if status != "SUCCESS":
            print(f"Worker {worker_id} Error: {status}")
            continue

        user_df = spark.read.schema(user_schema).parquet(file_path)
        filtered_df = user_df.filter(col("age") > 30).withColumn("worker_id", expr(f"{worker_id}"))
        processed_results.extend([row.asDict() for row in filtered_df.collect()])

    return {"users_over_30": processed_results, "metadata_cid": metadata_cid}
