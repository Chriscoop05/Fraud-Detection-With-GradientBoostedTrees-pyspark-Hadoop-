from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, when, sum, min, max
from pyspark.sql.functions import round as spark_round
import time

spark = SparkSession.builder \
    .appName("FraudAnalytics") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

t_start = time.time()
df = spark.read.csv(
    "hdfs://namenode:8020/user/fraud/raw/creditcard.csv",
    header=True,
    inferSchema=True
)
t_load = time.time() - t_start

print("rows loaded: "+str(df.count()))
print("load time: "+str(round(t_load, 2))+"s")
print("")

# class split -- shows how bad the imbalance actually is
print("QUERY 1: class distribution")
t0 = time.time()
df.groupBy("Class").agg(
    count("*").alias("count"),
    spark_round(avg("Amount"), 2).alias("avg_amount"),
    spark_round(min("Amount"), 2).alias("min_amount"),
    spark_round(max("Amount"), 2).alias("max_amount")
).orderBy("Class").show()
t1 = time.time() - t0
print("runtime: "+str(round(t1, 2))+"s\n")

# bucket transactions by dollar amount, see where fraud concentraits
print("QUERY 2: fraud rate by amount bracket")
t0 = time.time()
df.withColumn(
    "bracket",
    when(col("Amount") < 10, "Micro")
    .when(col("Amount") < 100, "Small")
    .when(col("Amount") < 500, "Medium")
    .when(col("Amount") < 1000, "Large")
    .otherwise("Very Large")
).groupBy("bracket") \
 .agg(
    count("*").alias("total"),
    sum(col("Class")).alias("fraud_count"),
    spark_round((sum(col("Class"))/count("*"))*100, 4).alias("fraud_rate_pct"),
    spark_round(avg("Amount"), 2).alias("avg_amount")
 ).orderBy("avg_amount").show()
t2 = time.time() - t0
print("runtime: "+str(round(t2, 2))+"s\n")

# Time column is seconds elapsed, dividing by 3600 gives hour buckets
print("QUERY 3: fraud by hour")
t0 = time.time()
df.withColumn("hour", (col("Time")/3600).cast("int")) \
  .groupBy("hour") \
  .agg(
    count("*").alias("total"),
    sum(col("Class")).alias("fraud_count"),
    spark_round((sum(col("Class"))/count("*"))*100, 4).alias("fraud_rate_pct")
  ).orderBy("hour").show(48)
t3 = time.time() - t0
print("runtime: "+str(round(t3, 2))+"s\n")

# compare average feature values between fraud and legit -- big differnces = predictive
print("QUERY 4: feature means by class (V1-V5)")
t0 = time.time()
print("fraud:")
df.filter(col("Class")==1).agg(
    spark_round(avg("V1"), 4).alias("V1"),
    spark_round(avg("V2"), 4).alias("V2"),
    spark_round(avg("V3"), 4).alias("V3"),
    spark_round(avg("V4"), 4).alias("V4"),
    spark_round(avg("V5"), 4).alias("V5"),
    spark_round(avg("Amount"), 2).alias("avg_amount")
).show()
print("legitimate:")
df.filter(col("Class")==0).agg(
    spark_round(avg("V1"), 4).alias("V1"),
    spark_round(avg("V2"), 4).alias("V2"),
    spark_round(avg("V3"), 4).alias("V3"),
    spark_round(avg("V4"), 4).alias("V4"),
    spark_round(avg("V5"), 4).alias("V5"),
    spark_round(avg("Amount"), 2).alias("avg_amount")
).show()
t4 = time.time() - t0
print("runtime: "+str(round(t4, 2))+"s\n")

# top 10 highest value fraud transactions
print("QUERY 5: top 10 fraud by amount")
t0 = time.time()
df.filter(col("Class")==1) \
  .select("Time", "Amount", "V1", "V2", "V3", "Class") \
  .orderBy(col("Amount").desc()) \
  .limit(10).show()
t5 = time.time() - t0
print("runtime: "+str(round(t5, 2))+"s\n")

print("total: "+str(round(t_load+t1+t2+t3+t4+t5, 2))+"s")

spark.stop()
