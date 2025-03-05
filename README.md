docker-compose build
docker-compose up -d

docker exec -it spark-worker-1 pip install pandas requests
docker exec -it spark-worker-2 pip install pandas requests
docker exec -it spark-worker-3 pip install pandas requests

Spark Master UI: http://localhost:8080/
Driver UI: http://localhost:8000/docs#/
