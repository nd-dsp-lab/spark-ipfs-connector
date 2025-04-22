# Web3db Worker Node Setup

### As of right now the master node is down and a live replica doesn't exist.

This will serve as a guide on how to setup a worker node to connect to the Web3db network.

## Prerequisites:

- Docker must be installed on a linux machine
  - For networking purposes

## Step 1

- Clone the repo by running

```
git clone https://github.com/nd-dsp-lab/spark-ipfs-connector.git
cd spark-ipfs-connector
```

- switch to the correct branch

```
git checkout verison1
```

## Step 2: Verify port availability

- The spark worker dynamically allocates ports, so as long as there are open ports, not being blocked by a firewall, the spark-worker should be able to connect assuming the master listed in worker.yml under SPARK_MASTER_URL is reachable (Should be a public ip or hostname). The spark-worker docker container uses the host network to bypass docker's networking isolation and connect to the spark-master node. This functionality isn't available on Mac OS, thus the requirement of linux.
- As configured, the ipfs node uses the following ports, which are exposed to the host in worker.yml, and configured within the ipfs/start-ipfs.sh script. If they aren't available, they should be changed in both places
  - 4002: ipfs swarm port
  - 5002: ipfs api gateway (ipfs get functionality)
    - represents a security concern when exposed to host because others can change ipfs configuration. Okay for development
  - 8082: ipfs gateway to facilitate node discovery

## Step 3: Running the worker node

- To run the worker node, run the following from the repo directory:

```
docker-compose -f worker.yml up --build -d
```

- This will build the ipfs image so that it includes any updates to the start-ipfs script, while also disconnecting the runtime from your terminal
- To shutdown, simply run the following:

```
docker-compose -f worker.yml down --volumes
```

## Other Information

- The start ipfs-script should include the most up-to-date master ipfs-node (currently down) so that the ipfs nodes can more-easily connect to the system. This may be out of date, and will slow down file transfer times
- Running `docker exec -it ipfs-private sh` will start a terminal within the ipfs node that can be used to test file transfers and peer connections
- Master Node setup info will come later. As of now, the master node requires a public IP or Spark to function
- Usage:
  - If the system is up, go to http://129.74.152.201:8000/docs#/ and try the post and then get function.
  - The functionality can be checked by going to http://129.74.152.201:8080 to view the spark job and executors.
