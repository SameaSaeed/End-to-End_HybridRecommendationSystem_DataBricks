# End-to-End Hybrid Recommendation System on Databricks

## Introduction

The End-to-End Hybrid Recommendation System leverages Databricks to build a scalable, flexible, and production-ready recommendation engine. This repository demonstrates how to preprocess data, train collaborative and content-based models, blend predictions, and deploy recommendations efficiently within the Databricks ecosystem. It is designed for online retail, e-commerce, or any domain requiring personalized item suggestions.

## Features

- End-to-end data pipeline: ingestion, preprocessing, feature engineering, model training, and prediction.
- Hybrid approach: blends collaborative filtering (ALS) and content-based filtering for improved accuracy.
- Scalable architecture using Apache Spark and Databricks.
- Model evaluation and hyperparameter tuning.
- Model deployment and batch/real-time inference support.
- Modular notebooks and scripts for extensibility.

## Requirements

- Databricks workspace (Community, Enterprise, or Azure/AWS Databricks)
- Databricks Runtime 8.1+ (with ML support recommended)
- Python 3.7+
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - pyspark
  - matplotlib (for visualization)
- Data: User-item interaction logs and item metadata (CSV, Parquet, or Delta format supported)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SameaSaeed/End-to-End_HybridRecommendationSystem_DataBricks.git
   cd End-to-End_HybridRecommendationSystem_DataBricks
   ```

2. **Import notebooks into Databricks:**
   - Go to your Databricks workspace.
   - Use the UI to import each notebook from the repository (`/notebooks` folder).

3. **Install required Python libraries:**
   - In a Databricks notebook cell or a cluster init script, run:
     ```python
     %pip install pandas numpy scikit-learn matplotlib
     ```

4. **Upload your data:**
   - Place your user-item interactions and item metadata files in Databricks File System (DBFS) or accessible cloud storage.

## Usage

1. **Data Ingestion and Preprocessing:**
   - Start with the `01_data_ingestion_preprocessing` notebook.
   - Configure the data paths and run all cells to clean and prepare your datasets.

2. **Feature Engineering:**
   - Open `02_feature_engineering`.
   - Generate user/item features and encode categorical variables.

3. **Collaborative Filtering (ALS):**
   - Use `03_collaborative_filtering_ALS` to train an ALS model.
   - Evaluate the model and save predictions.

4. **Content-Based Filtering:**
   - Proceed with `04_content_based_filtering`.
   - Train and evaluate a content-based recommender using item features.

5. **Hybrid Model Blending:**
   - The `05_hybrid_blending` notebook combines predictions from both models.
   - Tune blending ratios and evaluate hybrid performance.

6. **Model Deployment:**
   - Use `06_model_deployment` to save models and generate recommendations for new users.

7. **Batch or Real-Time Inference:**
   - Schedule inference notebooks as Databricks jobs for batch processing.
   - For real-time, deploy models as Databricks ML endpoints (Databricks MLflow support required).

### Example: Running the Pipeline

```python
# Import necessary Spark and ML libraries
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession

# Load data
interactions = spark.read.csv("/dbfs/data/interactions.csv", header=True, inferSchema=True)
metadata = spark.read.csv("/dbfs/data/items.csv", header=True, inferSchema=True)

# Train ALS model
als = ALS(userCol="user_id", itemCol="item_id", ratingCol="rating", coldStartStrategy="drop")
model = als.fit(interactions)

# Generate recommendations
recommendations = model.recommendForAllUsers(10)
recommendations.show()
```

## Configuration

- **Data Paths:** Configure your input and output data paths in the first cell of each notebook.
- **Model Parameters:** Adjust ALS hyperparameters (rank, regParam, maxIter) and content-based model settings in the respective notebooks.
- **Blending Ratio:** Set the weight for collaborative vs. content-based predictions in the hybrid blending notebook.
- **Cluster Size:** Scale your Databricks cluster based on data size and computational needs.

### Example Configuration Cell

```python
# Set paths and parameters
INTERACTIONS_PATH = "/dbfs/data/interactions.csv"
METADATA_PATH = "/dbfs/data/items.csv"
ALS_RANK = 10
ALS_REGPARAM = 0.1
BLEND_RATIO = 0.7  # 70% collaborative, 30% content-based
```

## Contributing

Contributions are welcome! To get involved:

- Fork the repository and create a new branch for your feature or fix.
- Write clear and concise code, adhering to PEP8 guidelines.
- Document your code and update the relevant sections of the README if necessary.
- Submit a pull request with a detailed description.

**Development guidelines:**
- Use modular functions and classes.
- Write reproducible notebooks (restart and run all cells before submitting).
- Add meaningful comments and docstrings.
- Include tests for new modules where applicable.

---

```card
{
  "title": "Contribution Tip",
  "content": "Test your notebooks end-to-end in a new Databricks workspace to ensure reproducibility before submitting a pull request."
}
```

---

For questions, open an issue or contact the repository maintainer via GitHub.