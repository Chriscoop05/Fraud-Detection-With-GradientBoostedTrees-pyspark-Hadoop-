from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import time

spark = SparkSession.builder \
    .appName("FraudDetection") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

t_start = time.time()
df = spark.read.csv(
    "hdfs://namenode:8020/user/fraud/raw/creditcard.csv",
    header=True,
    inferSchema=True
)
t_load = time.time() - t_start
print("loaded "+str(df.count())+" rows in "+str(round(t_load, 2))+"s\n")

# only 492 fraud out of 284k rows so we weight them way higher
total = df.count()
fraud_count = df.filter(col("Class")==1).count()
legit_count = df.filter(col("Class")==0).count()

fraud_weight = total/(2.0*fraud_count)
legit_weight = total/(2.0*legit_count)

print("fraud: "+str(fraud_count)+" ("+str(round(fraud_count/total*100, 4))+"%)")
print("fraud weight: "+str(round(fraud_weight, 2)))
print("legit weight: "+str(round(legit_weight, 4))+"\n")

df = df.withColumn(
    "classWeight",
    when(col("Class")==1, fraud_weight).otherwise(legit_weight)
)

# V1-V28 already scaled from PCA, Amount and Time are raw so we normalize them
feature_cols = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
    "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18",
    "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27",
    "V28", "Amount"
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True)

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
print("train: "+str(train_df.count())+" | test: "+str(test_df.count())+"\n")

gbt = GBTClassifier(
    labelCol="Class",
    featuresCol="features",
    weightCol="classWeight",
    maxIter=50,
    maxDepth=5,
    stepSize=0.1,
    seed=42
)

pipeline = Pipeline(stages=[assembler, scaler, gbt])

t0 = time.time()
model = pipeline.fit(train_df)
t_train = time.time() - t0
print("training done in "+str(round(t_train, 2))+"s\n")

predictions = model.transform(test_df)

print("sample fraud predictions:")
predictions.filter(col("Class")==1).select("Class", "prediction", "Amount").show(10)

roc_evaluator = BinaryClassificationEvaluator(labelCol="Class", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
pr_evaluator = BinaryClassificationEvaluator(labelCol="Class", rawPredictionCol="rawPrediction", metricName="areaUnderPR")
precision_evaluator = MulticlassClassificationEvaluator(labelCol="Class", predictionCol="prediction", metricName="weightedPrecision")
recall_evaluator = MulticlassClassificationEvaluator(labelCol="Class", predictionCol="prediction", metricName="weightedRecall")
f1_evaluator = MulticlassClassificationEvaluator(labelCol="Class", predictionCol="prediction", metricName="f1")

roc_auc = roc_evaluator.evaluate(predictions)
pr_auc = pr_evaluator.evaluate(predictions)
precision = precision_evaluator.evaluate(predictions)
recall = recall_evaluator.evaluate(predictions)
f1 = f1_evaluator.evaluate(predictions)

tp = predictions.filter((col("Class")==1)&(col("prediction")==1)).count()
tn = predictions.filter((col("Class")==0)&(col("prediction")==0)).count()
fp = predictions.filter((col("Class")==0)&(col("prediction")==1)).count()
fn = predictions.filter((col("Class")==1)&(col("prediction")==0)).count()

print("confusion matrix:")
print("TP (caught fraud):  "+str(tp))
print("TN (correct legit): "+str(tn))
print("FP (false alarms):  "+str(fp))
print("FN (missed fraud):  "+str(fn))
print("")
print("ROC-AUC:   "+str(round(roc_auc, 4)))
print("PR-AUC:    "+str(round(pr_auc, 4)))
print("Precision: "+str(round(precision, 4)))
print("Recall:    "+str(round(recall, 4)))
print("F1:        "+str(round(f1, 4)))
print("")

# which features the model leaned on most
gbt_model = model.stages[-1]
importances = gbt_model.featureImportances.toArray()
feature_importance = list(zip(feature_cols, importances))
feature_importance.sort(key=lambda x: x[1], reverse=True)

print("top 10 features:")
for i, (feat, imp) in enumerate(feature_importance[:10]):
    print(str(i+1)+". "+feat+" -- "+str(round(imp, 4)))

print("")
print("training time: "+str(round(t_train, 2))+"s")
print("ROC-AUC: "+str(round(roc_auc, 4)))
print("PR-AUC: "+str(round(pr_auc, 4)))
print("fraud caught: "+str(tp)+" / "+str(tp+fn))
print("false alarms: "+str(fp))

spark.stop()
