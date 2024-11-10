import argparse
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StandardScaler, StopWordsRemover
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col, when, trim, lower
from datetime import datetime
import os

base_model_path = "../app/model"

def main(file_path):
    spark = SparkSession.builder \
        .appName("Toxic Comment Classifier") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    
    df = spark.read.option("quote", "\"") \
        .option("escape", "\"") \
        .option("multiline", "true") \
        .csv(file_path, header=True, inferSchema=True)

    df = df.withColumn("toxicity_level", 
        (col("toxic").cast("double") + 
        col("severe_toxic").cast("double") + 
        col("obscene").cast("double") + 
        col("threat").cast("double") + 
        col("insult").cast("double") + 
        col("identity_hate").cast("double")))

    df = df.withColumn("comment_text", 
        when(col("comment_text").isNull(), "") 
        .otherwise(trim(lower(col("comment_text")))))  

    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

    tokenizer = Tokenizer(inputCol="comment_text", 
                      outputCol="words")

    stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")

    hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)

    idf = IDF(inputCol="raw_features", outputCol="features")

    scaler = StandardScaler(inputCol="features", 
                    outputCol="scaled",
                    withStd=True, 
                    withMean=False)

    lr = LogisticRegression(labelCol="toxicity_level", 
                    featuresCol="features", 
                    maxIter=10)
    
    pipelines = {
    "Logistic_Regression": Pipeline(stages=[tokenizer, stopwords_remover, hashingTF, idf, lr]),
    }

    for name, pipeline in pipelines.items():
        print(f"\nTraining {name}...")
        
        # Fit the pipeline
        model = pipeline.fit(train_data)
        
        # Make predictions
        predictions = model.transform(test_data)
        
        # Select example rows to display
        predictions.select("comment_text", "toxicity_level", "prediction").show(5, truncate=False)
        
        # Evaluate the model
        evaluator_accuracy = MulticlassClassificationEvaluator(
            labelCol="toxicity_level", 
            predictionCol="prediction", 
            metricName="accuracy"
        )
        
        evaluator_f1 = MulticlassClassificationEvaluator(
            labelCol="toxicity_level", 
            predictionCol="prediction", 
            metricName="f1"
        )
        
        accuracy = evaluator_accuracy.evaluate(predictions)
        f1 = evaluator_f1.evaluate(predictions)
        
        print(f"{name} Results:")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"F1 Score: {f1:.2f}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(base_model_path, f"{name}_{timestamp}")
        model.save(model_path)
        print(f"\nModel saved to: {model_path}")
        
        # Save example of how to load the model in a README file
        readme_path = os.path.join(model_path, "README.txt")
        with open(readme_path, "w") as f:
            f.write(f"""Model: {name}
Saved on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Accuracy: {accuracy:.4f}
F1 Score: {f1:.4f}"""
                    )

    # Stop Spark session
    spark.stop()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('-f', '--file', required=True, help='Path to the file')
    args = parser.parse_args()
    main(args.file)
