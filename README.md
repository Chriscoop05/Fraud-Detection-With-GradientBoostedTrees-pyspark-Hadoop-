# Distributed Credit Card Fraud Detection
### Apache Spark MLlib · Hadoop HDFS · Docker · Python

**Dataset:** [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) -- download `creditcard.csv` and place it in the `data/` folder before running any scripts

---

## Overview

Built an end-to-end distributed machine learning pipeline for credit card fraud detection using Apache Spark MLlib on a multi-container Hadoop/Spark cluster deployed via Docker. The project covers the full data engineering lifecycle -- from distributed storage in HDFS through exploratory analytics, model training, hyperparameter optimization, and scalability analysis across increasing data volumes.

The core challenge this project addresses is extreme class imbalance: only 0.17% of transactions in the dataset are fraudulent. Standard machine learning approaches fail on datasets like this because a model that always predicts "legitimate" achieves 99.83% accuracy while catching zero fraud. This project implements several techniques to overcome that challenge and achieve a final PR-AUC of 0.9885 with zero missed fraud cases.

---

## Technical Stack

- **Apache Spark 3.5.7** -- distributed data processing and MLlib pipeline
- **Hadoop 3.2.1 / HDFS** -- distributed file storage with 3x replication
- **Docker** -- multi-container cluster orchestration (11 containers across 2 compose clusters)
- **Python / PySpark** -- data engineering, feature engineering, model training
- **Gradient Boosted Trees** -- primary classification algorithm via Spark MLlib

---

## Dataset

**Credit Card Fraud Detection Dataset** (Kaggle / ULB Machine Learning Group)

- 284,807 transactions, 31 features, 151MB
- Target: `Class` (0 = legitimate, 1 = fraud)
- 492 fraud cases -- 0.17% of all transactions
- Features V1-V28: PCA-transformed behavioral features (anonymized)
- Raw features: `Time` (seconds elapsed), `Amount` (transaction value in USD)
- No missing values

---

## Architecture

Two separate Docker clusters bridged together via a dual-network gateway node:

```
Hadoop Cluster                          Spark Cluster
(docker-hadoop_default network)         (docker-spark_spark-net)

namenode        -- HDFS metadata        spark-master  -- cluster manager
datanode-1      -- data blocks          spark-worker  -- executor (2 cores, 2GB)
datanode-2      -- replica blocks       spark-client  -- driver + network bridge
nodemanager-1   -- YARN tasks
nodemanager-2   -- YARN tasks
resourcemanager -- job coordination
```

`spark-client` connects to both networks and acts as the gateway between the storage layer (Hadoop) and the compute layer (Spark). `spark-worker` connects directly to the Hadoop network to stream data blocks from the datanodes during model training, bypassing the driver for data transfer.

Static IP assignment on the Spark network ensures stable hostname resolution across container restarts without manual network reconfiguration.

The `spark-client` container is built from `Dockerfile.client` rather than the standard Spark image. It extends `spark:3.5.7-python3` with `wget`, `pip`, `pyspark`, and `pandas` to support the full analytics and ML workload. The build happens automatically when you run `docker-compose up -d` for the first time.

---

## Reproducing This Project

### Prerequisites

- Docker Desktop
- A running Hadoop cluster (the Spark compose file expects a Docker network named `docker-hadoop_default` to already exist)
- The `creditcard.csv` dataset downloaded from Kaggle

### Setup

**1. Clone the repo:**
```bash
git clone https://github.com/yourusername/credit-card-fraud-detection
cd credit-card-fraud-detection
```

**2. Place the dataset in the data folder:**
```bash
cp ~/Downloads/creditcard.csv data/creditcard.csv
```

The `data/` folder maps to `/opt/data/` inside the Spark containers via the volume mount in `docker-compose.yml`. All scripts read from this location.

**3. Start the Spark cluster:**
```bash
docker-compose up -d
```

This builds the `spark-client` image from `Dockerfile.client` on first run and starts all three Spark containers. Make sure your Hadoop cluster is already running before this step.

**4. Upload data to HDFS:**
```bash
docker cp data/creditcard.csv docker-hadoop-namenode-1:/tmp/creditcard.csv
docker exec -i docker-hadoop-namenode-1 hdfs dfs -mkdir -p /user/fraud/raw
docker exec -i docker-hadoop-namenode-1 hdfs dfs -put -f /tmp/creditcard.csv /user/fraud/raw/creditcard.csv
```

**5. Run the scripts in order:**
```bash
# exploratory analysis
docker exec -it spark-client /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  /opt/data/fraud_analytics.py

# baseline model
docker exec -it spark-client /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  /opt/data/fraud_model.py

# improved model (takes ~60 min due to cross validation)
docker exec -it spark-client /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  /opt/data/fraud_model_v2.py

# scalability analysis
docker exec -it spark-client /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  /opt/data/fraud_scalability.py
```

---

## Pipeline

### 1. Data Ingestion & HDFS Storage

Loaded the raw CSV into HDFS with 3x replication across two datanodes. Created a partitioned directory structure:

```
/user/fraud/
raw/creditcard.csv          -- 151MB, 3 replicas
processed/                  -- preprocessed output
mapreduce_output/           -- aggregation results
```

Verified data integrity with `hdfs dfsadmin -report` confirming block distribution across both datanodes.

### 2. Exploratory Analytics (Spark SQL)

Ran five analytical queries against the raw HDFS data to understand the fraud patterns before modeling:

**Class distribution** -- confirmed extreme imbalance: 492 fraud vs 284,315 legitimate transactions. Average fraud transaction ($122) was higher than legitimate ($88), suggesting fraudsters target higher value transactions.

**Amount bracket analysis** -- fraud rate varied significantly across transaction sizes, with mid-range transactions ($100-$500) showing disproportionate fraud concentration.

**Time-based patterns** -- fraud frequency showed temporal clustering, with elevated rates during certain hour windows suggesting coordinated fraud activity.

**Feature comparison** -- V1, V3, V4, V9, V10, V11, V14 showed the largest mean differences between fraud and legitimate transactions, flagging them as likely high-importance features before any model training.

**Top fraud transactions** -- all 10 highest-value fraud transactions exceeded $1,000, with the largest at $2,125.

Total query runtime: under 60 seconds on the distributed cluster for 284,807 rows.

### 3. Baseline Model (V1)

Trained an initial Gradient Boosted Trees classifier with class weighting to handle the imbalance. Class weights were computed as `total / (2 * class_count)`, assigning a weight of ~289 to each fraud transaction vs ~0.5 to each legitimate transaction.

**V1 Results:**
- ROC-AUC: 0.9632
- PR-AUC: 0.5346
- Fraud caught: 79 out of 98 test cases (80.6%)
- False alarms: 251

The ROC-AUC was strong but PR-AUC of 0.5346 indicated significant room for improvement on the fraud-specific detection task. Feature importance showed V14 accounting for 56% of the model's decisions -- a single PCA feature dominating the classifier.

### 4. Improved Model (V2)

Implemented five targeted improvements to address the V1 limitations:

**Feature Engineering**
- Log-transformed `Amount` (`log(Amount + 1)`) to compress the extreme right skew from $0 to $25,691
- Added interaction terms between the three most important V1 features: `V14×V4`, `V14×V12`, `V4×V12`, and `V14²` to capture nonlinear relationships the base features couldn't express

**SMOTE-Style Oversampling**
Duplicated fraud cases 10x before training, increasing fraud representation from 0.17% to approximately 1.7%. Combined with class weighting this gave the model substantially more fraud signal to learn from without discarding any legitimate transaction data.

**Hyperparameter Tuning via Cross Validation**
Ran 3-fold cross validation across 8 parameter combinations using PR-AUC as the optimization metric:
- `maxIter`: [50, 100]
- `maxDepth`: [4, 6]
- `stepSize`: [0.05, 0.1]

Used PR-AUC rather than ROC-AUC as the CV metric since PR-AUC is more meaningful for imbalanced classification -- it focuses specifically on precision and recall for the minority (fraud) class rather than overall ranking ability.

**V2 Results:**
- ROC-AUC: 0.9999
- PR-AUC: 0.9885
- True Positives: 994 (fraud correctly caught)
- True Negatives: 56,732 (legitimate correctly approved)
- False Positives: 134 (legitimate flagged as fraud)
- False Negatives: 0 (fraud missed)
- Training time: 3,770 seconds (cross validation, 24 total model fits)

**Zero missed fraud cases** at the default 0.5 decision threshold -- the combination of oversampling, interaction features, and tuned hyperparameters produced a near-perfect classifier on this dataset.

### 5. Scalability Analysis

Trained the fraud detection pipeline on 1x, 2x, 5x, and 10x replications of the base dataset to measure how the distributed system handles increasing data volume:

| Scale | Rows | Train Time | ROC-AUC |
|---|---|---|---|
| 1x | 284,807 | 31.0s | 0.9657 |
| 2x | 569,614 | 38.7s | 0.9916 |
| 5x | 1,424,035 | 85.5s | 0.9957 |
| 10x | 2,848,070 | 336.0s | 0.9960 |

Two findings stood out. First, model quality improved consistently with more data -- ROC-AUC increased by 3 points from 1x to 2x, demonstrating that fraud detection models benefit meaningfully from larger training sets since rare fraud examples become more represented. Second, training time scaled sublinearly up to 5x (5x data, 2.75x time) before hitting memory pressure at 10x where the single 2GB executor began spilling to disk, causing a disproportionate time increase. In a production cluster with multiple workers and larger memory allocation the 10x result would scale much more efficiently.

---

## Key Results

| | Baseline (V1) | Improved (V2) |
|---|---|---|
| ROC-AUC | 0.9632 | 0.9999 |
| PR-AUC | 0.5346 | 0.9885 |
| Fraud caught | 79 / 98 (80.6%) | 994 / 994 (100%) |
| False alarms | 251 | 134 |
| Features | 30 | 35 (+ engineered) |

---

## Technical Challenges

**Network isolation between clusters** -- Spark and Hadoop ran on separate Docker bridge networks with no default routing between them. Solved by connecting `spark-client` to both networks as a gateway node and assigning static IPs to Spark containers to prevent hostname resolution conflicts caused by Docker's dynamic IP assignment across container restarts.

**Extreme class imbalance** -- 0.17% fraud rate made standard accuracy metrics misleading. Addressed through class weighting, oversampling, and using PR-AUC as both the optimization target during cross validation and the primary evaluation metric.

**Sparse vector indexing** -- Spark's cross validator caused the probability output column to be stored as a sparse vector rather than a dense array, breaking standard bracket indexing syntax. Resolved by switching to Spark's native `.getItem()` method for vector element extraction.

**Scalability wall at 10x** -- single executor memory ceiling of 2GB caused disk spilling at 2.8M rows. Production mitigation would be horizontal scaling with additional worker nodes rather than vertical memory scaling.

---

## Repository Structure

```
credit-card-fraud-detection/
README.md
.gitignore
requirements.txt
docker-compose.yml
Dockerfile.client
fraud_analytics.py       -- Spark SQL exploratory analysis (5 queries)
fraud_model.py           -- Baseline GBT model with class weighting
fraud_model_v2.py        -- Improved model with all enhancements
fraud_scalability.py     -- Scalability experiment (1x to 10x)
 data/
    .gitkeep             -- keeps the data/ folder tracked by Git
 results/
     v1_evaluation.png
     v2_evaluation.png
     scalability_table.png
     spark_ui.png
```

---

## What I'd Do Next

- **True SMOTE** using KNN-based synthetic sample generation rather than simple duplication
- **Horizontal scaling** -- add additional Spark workers to eliminate the memory bottleneck at large data sizes
- **Drift detection** -- simulate concept drift by shifting the Amount distribution over time and measure model degradation

