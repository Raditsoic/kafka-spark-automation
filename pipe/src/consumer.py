from confluent_kafka import Consumer, KafkaError
import json
import pandas as pd

conf = {
    'bootstrap.servers': 'localhost:9092',
    'auto.offset.reset': 'earliest',
    'group.id': 'python-consumer-group',
}
consumer = Consumer(conf)

topic = 'comment'  
consumer.subscribe([topic])

batch_size = 53190  
batch_data = []

header_written = False
batch_number = 1

print("Booting Up Kafka Consumer...")

try:
    while True:
        msg = consumer.poll(timeout=1.0)
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                print(f"End of partition reached {msg.topic()} [{msg.partition()}] at offset {msg.offset()}")
            else:
                print(f"Error: {msg.error()}")
            continue

        message = json.loads(msg.value().decode('utf-8'))
        batch_data.append(message)

        print(f"Message {msg.key()} added to batch")

        if len(batch_data) >= (batch_size * batch_number):
            df = pd.DataFrame(batch_data)

            csv_file_path = f'../dataset/batch/toxic_comment_{batch_number}.csv'

            df.to_csv(csv_file_path, index=False)
            print(f"Batch {batch_number} saved to {csv_file_path}")

            print(f"Processing batch of {batch_size} messages")
            print(df.head()) 
            print(f"Batch of {batch_size} messages saved to {csv_file_path}")

            batch_number += 1

except KeyboardInterrupt:
    pass
finally:
    csv_file_path = './dataset/batch/toxic_comment_final.csv'
    if batch_data:
        df = pd.DataFrame(batch_data)
        df.to_csv(csv_file_path, mode='a', index=False, header=not header_written)
        print(f"Final batch of {len(batch_data)} messages saved to {csv_file_path}")
    
    consumer.close()
