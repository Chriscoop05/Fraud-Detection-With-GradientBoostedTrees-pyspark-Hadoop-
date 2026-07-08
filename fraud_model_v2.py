from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, log, lit
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import time

spark = SparkSession.builder \
    .appName("FraudDetectionV2") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

t_start = time.time()
df = spark.read.csv(
    "hdfs://namenode:8020/user/fraud/raw/creditcard.csv",
    header=True,
    inferSchema=True
)
total = df.count()
fraud_count = df.filter(col("Class")==1).count()
legit_count = df.filter(col("Class")==0).count()
t_load = time.time() - t_start

print("loaded "+str(total)+" rows in "+str(round(t_load, 2))+"s")
print("fraud: "+str(fraud_count)+" | legit: "+str(legit_count)+"\n")

# log transform compresses the $0-$25k range, +1 avoids log(0) on zero dollar txns
df = df.withColumn("Amount_log", log(col("Amount")+lit(1)))

# V14 was 56% of v1 model importance so interactions with it should help
df = df.withColumn("V14_x_V4", col("V14")*col("V4"))
df = df.withColumn("V14_x_V12", col("V14")*col("V12"))
df = df.withColumn("V4_x_V12", col("V4")*col("V12"))
df = df.withColumn("V14_squared", col("V14")*col("V14"))

print("engineered features: Amount_log, V14_x_V4, V14_x_V12, V4_x_V12, V14_squared\n")

# oversample fraud 10x so model sees enough positive examples to learn from
fraud_df = df.filter(col("Class")==1)
legit_df = df.filter(col("Class")==0)

oversampled_fraud = fraud_df
for _ in range(9):
    oversampled_fraud = oversampled_fraud.union(fraud_df)

df_balanced = legit_df.union(oversampled_fraud)

new_fraud = df_balanced.filter(col("Class")==1).count()
new_legit = df_balanced.filter(col("Class")==0).count()
new_total = df_balanced.count()

print("before oversampling -- fraud: "+str(fraud_count)+" legit: "+str(legit_count))
print("after oversampling  -- fraud: "+str(new_fraud)+" legit: "+str(new_legit))
print("new fraud rate: "+str(round(new_fraud/new_total*100, 2))+"%\n")

fraud_weight = new_total/(2.0*new_fraud)
legit_weight = new_total/(2.0*new_legit)

df_balanced = df_balanced.withColumn(
    "classWeight",
    when(col("Class")==1, fraud_weight).otherwise(legit_weight)
)

train_df, test_df = df_balanced.randomSplit([0.8, 0.2], seed=42)
print("train: "+str(train_df.count())+" | test: "+str(test_df.count())+"\n")

feature_cols = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
    "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18",
    "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27",
    "V28", "Amount_log",
    "V14_x_V4", "V14_x_V12", "V4_x_V12", "V14_squared"
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features", handleInvalid="skip")
scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True)
gbt = GBTClassifier(labelCol="Class", featuresCol="features", weightCol="classWeight", seed=42)
pipeline = Pipeline(stages=[assembler, scaler, gbt])

# PR-AUC is better than ROC-AUC for imbalanced problems, use it as CV metric
pr_evaluator = BinaryClassificationEvaluator(
    labelCol="Class",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderPR"
)

param_grid = ParamGridBuilder() \
    .addGrid(gbt.maxIter, [50, 100]) \
    .addGrid(gbt.maxDepth, [4, 6]) \
    .addGrid(gbt.stepSize, [0.05, 0.1]) \
    .build()

cv = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=param_grid,
    evaluator=pr_evaluator,
    numFolds=3,
    seed=42
)

print("running cross validation -- 8 param combos x 3 folds, this takes a while...")
t0 = time.time()
cv_model = cv.fit(train_df)
t_cv = time.time() - t0

best_gbt = cv_model.bestModel.stages[-1]
print("cv done in "+str(round(t_cv, 2))+"s")
print("best params -- maxIter: "+str(best_gbt.getMaxIter())+" maxDepth: "+str(best_gbt.getMaxDepth())+" stepSize: "+str(best_gbt.getStepSize())+"\n")

predictions = cv_model.transform(test_df)

roc_evaluator = BinaryClassificationEvaluator(labelCol="Class", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
roc_auc = roc_evaluator.evaluate(predictions)
pr_auc = pr_evaluator.evaluate(predictions)

tp = predictions.filter((col("Class")==1)&(col("prediction")==1)).count()
tn = predictions.filter((col("Class")==0)&(col("prediction")==0)).count()
fp = predictions.filter((col("Class")==0)&(col("prediction")==1)).count()
fn = predictions.filter((col("Class")==1)&(col("prediction")==0)).count()

print("results at default threshold 0.5:")
print("TP (caught fraud):  "+str(tp))
print("TN (correct legit): "+str(tn))
print("FP (false alarms):  "+str(fp))
print("FN (missed fraud):  "+str(fn))
print("ROC-AUC: "+str(round(roc_auc, 4)))
print("PR-AUC:  "+str(round(pr_auc, 4))+"\n")

# test different cutoffs to see trade off between catching fraud and false alarms
print("threshold tuning:")
print("threshold  TP   FP   FN   fraud_caught%")
print("-"*50)

best_threshold = 0.5
best_f1 = 0.0

for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
    preds_t = predictions.withColumn(
        "prediction",
        when(col("probability").getItem(1)>=threshold, 1.0).otherwise(0.0)
    )
    tp_t = preds_t.filter((col("Class")==1)&(col("prediction")==1)).count()
    fp_t = preds_t.filter((col("Class")==0)&(col("prediction")==1)).count()
    fn_t = preds_t.filter((col("Class")==1)&(col("prediction")==0)).count()
    caught_pct = round(tp_t/(tp_t+fn_t)*100, 1) if (tp_t+fn_t) > 0 else 0
    p = tp_t/(tp_t+fp_t) if (tp_t+fp_t) > 0 else 0
    r = tp_t/(tp_t+fn_t) if (tp_t+fn_t) > 0 else 0
    f1 = 2*p*r/(p+r) if (p+r) > 0 else 0
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
    print(str(threshold)+"    "+str(tp_t)+"   "+str(fp_t)+"   "+str(fn_t)+"   "+str(caught_pct)+"%")

print("\nbest threshold: "+str(best_threshold)+"\n")

final_preds = predictions.withColumn(
    "prediction",
    when(col("probability").getItem(1)>=best_threshold, 1.0).otherwise(0.0)
)

tp_f = final_preds.filter((col("Class")==1)&(col("prediction")==1)).count()
tn_f = final_preds.filter((col("Class")==0)&(col("prediction")==0)).count()
fp_f = final_preds.filter((col("Class")==0)&(col("prediction")==1)).count()
fn_f = final_preds.filter((col("Class")==1)&(col("prediction")==0)).count()

print("final results at threshold "+str(best_threshold)+":")
print("TP: "+str(tp_f)+" | TN: "+str(tn_f)+" | FP: "+str(fp_f)+" | FN: "+str(fn_f))
print("fraud detection rate: "+str(round(tp_f/(tp_f+fn_f)*100, 1))+"%")
print("false alarm rate: "+str(round(fp_f/(fp_f+tn_f)*100, 4))+"%")
print("ROC-AUC: "+str(round(roc_auc, 4)))
print("PR-AUC:  "+str(round(pr_auc, 4))+"\n")

# feature importances from the winning model
best_gbt_model = cv_model.bestModel.stages[-1]
importances = best_gbt_model.featureImportances.toArray()
feature_importance = list(zip(feature_cols, importances))
feature_importance.sort(key=lambda x: x[1], reverse=True)

print("top 10 features:")
for i, (feat, imp) in enumerate(feature_importance[:10]):
    print(str(i+1)+". "+feat+" -- "+str(round(imp, 4)))

spark.stop()
