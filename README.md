## **Apache Kafka - Apache Spark Flow**

| Name | NRP |
|---|---|
| Awang Fraditya | 5027221055 |
| Nur Azka Rahadiansyah | 502722106 |

In this app, we don't use Spark Stream, but we pipe the producer to the server and we pipe it again to the consumer. The consumer will batch and save the data in batches. It will trigger a cron/inotify that will train with Pyspark ML and outputs a model. The model will be used as an API to predict wether a comment is toxic or not and scale the level of toxicity.