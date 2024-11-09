from confluent_kafka import Producer
import json
import pandas as pd
import time

# Producer
conf = {
    'bootstrap.servers': 'localhost:9092',
    'client.id': 'python-producer',
    'acks': 'all',  
    'retries': 3,  
    'message.send.max.retries': 3,
    'queue.buffering.max.messages': 1000000,
    }

producer = Producer(conf)
print("Spinning up the Kafka Producer...")

def delivery_report(err, msg):
    if err is not None:
        print(f'Message delivery failed: {err}')
    else:
        print(f'Message delivered to {msg.topic()} [{msg.partition()}]')


csv_file = '../dataset/toxic_comment.csv'
data = pd.read_csv(csv_file)

topic = 'comment'

for index, row in data.iterrows():
    message = {
        "comment_text": row["comment_text"],
        "toxic": row["toxic"],
        "severe_toxic": row["severe_toxic"],
        "obscene": row["obscene"],
        "threat": row["threat"],
        "insult": row["insult"],
        "identity_hate": row["identity_hate"]
    }

    producer.produce(topic, key=str(index), value=json.dumps(message), callback=delivery_report)

    print(f"Message {index} sent to Kafka")

    # Uncomment if need delay and adjustment
    # time.sleep(0.01)


producer.flush()
print("All messages sent.")