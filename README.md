## **Apache Kafka - Apache Spark Flow**

| Name | NRP |
|---|---|
| Awang Fraditya | 5027221055 |
| Nur Azka Rahadiansyah | 502722106 |

In this app, we don't use Spark Stream, but we pipe the producer to the server and we pipe it again to the consumer. The consumer will batch and save the data in batches. It will trigger a cron/inotify that will train with Pyspark ML and outputs a model. The model will be used as an API to predict wether a comment is toxic or not and scale the level of toxicity.

## **Architecture**

![Architecture](/img/architecture.png)

- Kafka Producer will read csv file and stream it to Kafka Server with `comment` topic.
- Kafka Consumer will subscribe to `comment` topic on Kafka Server, Kafka Consumer will batch and save it in `.csv` format.
- While Kafka Consumer saving file in `batch` directory, cron will monitor the directory and trigger PySpark ML Trainer and train latest batched file.
- FastAPI will load the trained file and deploy in production (No CI/CD yet)

## **How to run it?**

### **Prerequisites**

- Docker
- Python 
- Linux

### **Clone this project**

`git clone https://github.com/Raditsoic/kafka-spark-automation.git`

### **Run Kafka**

Run Kafka-Zookeeper in Dockerized Environment
`cd kafka-spark-automation/pipe && docker compose up --build`

### **Make Topic**

To make topic we have to `exec` inside the kafka docker shell
`docker ps`

Copy the kafka id and put it here
`docker exec -it <kafka-id> /opt/kafka/bin/kafka-topics.sh --create --zookeeper zookeeper:2181 --replication-factor 1 --partitions 1 --topic comment`

### **Start the automation**

The automation using cron and bash, so we need to use this command
`bash automation.sh`

### **Install Consumer, Producer, and Trainer Depedencies**

`cd src && pip install -r requirements.txt`

### **Run the Consumer & Producer**

`python consumer.py` & `python producer.py`

### **Run & Build the API**

Build the API with docker
`cd ../app && docker build -t toxic-comment-classifier .`

Run the API
`docker run -p 8000:8000 toxic-comment-classifier`

### **Make Request to the API Endpoints**

#### Health Check
```sh
curl --location 'http://localhost:8000/health-check'
```

#### Prediction
```sh
curl --location 'http://localhost:8000/predict/v1.1' \
--header 'Content-Type: application/json' \
--data '{
  "text": "Hello I am You~~
}'
```

####  Batch Prediction
```sh
curl --location 'http://localhost:8000/predict/v1.1/batch' \
--header 'Content-Type: application/json' \
--data '{
  "comments": [
    "I am happy.",
    "This is bad words~",
    "Hi! I am back again! Last warning! Stop undoing my edits or die!"
  ]
}'
```