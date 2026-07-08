from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import time

spark = SparkSession.builder \
    .appName("FraudScalability") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

base_df = spark.read.csv(
    "hdfs://namenode:8020/user/fraud/raw/creditcard.csv",
    header=True,
    inferSchema=True
)

base_count = base_df.count()
print("base dataset: "+str(base_count)+" rows\n")

total = base_count
fraud_count = base_df.filter(col("Class")==1).count()
legit_count = base_df.filter(col("Class")==0).count()
fraud_weight = total/(2.0*fraud_count)
legit_weight = total/(2.0*legit_count)

base_df = base_df.withColumn(
    "classWeight",
    when(col("Class")==1, fraud_weight).otherwise(legit_weight)
)

feature_cols = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
    "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18",
    "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27",
    "V28", "Amount"
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True)
gbt = GBTClassifier(labelCol="Class", featuresCol="features", weightCol="classWeight", maxIter=20, maxDepth=4, seed=42)
pipeline = Pipeline(stages=[assembler, scaler, gbt])
evaluator = BinaryClassificationEvaluator(labelCol="Class", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

results = []

print("multiplier  rows       train_time  roc_auc")
print("-"*50)

for mult in [1, 2, 5, 10]:
    scaled_df = base_df
    for _ in range(mult-1):
        scaled_df = scaled_df.union(base_df)

    row_count = base_count*mult
    train_df, test_df = scaled_df.randomSplit([0.8, 0.2], seed=42)

    t0 = time.time()
    model = pipeline.fit(train_df)
    t_train = time.time() - t0

    preds = model.transform(test_df)
    roc = evaluator.evaluate(preds)

    results.append((mult, row_count, t_train, roc))
    print(str(mult)+"x  "+str(row_count)+"  "+str(round(t_train, 2))+"s  "+str(round(roc, 4)))

print("")
for mult, rows, t, roc in results:
    print(str(mult)+"x  "+str(rows)+"  "+str(round(t, 2))+"s  "+str(round(roc, 4)))

spark.stop()
