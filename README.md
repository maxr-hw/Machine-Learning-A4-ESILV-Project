# Electronic Sales Clustering & Analysis

## Project Overview

This project analyzes a dataset of electronic product sales and explores **customer purchase behavior** using unsupervised learning and clustering. The goal is to:

1. Explore and visualize transaction data.  
2. Segment transactions/customers into **clusters** based on purchasing behavior.  
3. Enable prediction of cluster membership for new transactions and summarize cluster characteristics.  
4. Provide interpretable insights on quantity, total sale, and customer/product patterns.

---

## Dataset

- File: `Electronic_sales_Sep2023-Sep2024.csv`  from `https://www.kaggle.com/datasets/cameronseamons/electronic-sales-sep2023-sep2024`
- Key columns include:
  - Customer information: `Customer ID`, `Age`, `Gender`, `Loyalty Member`
  - Transaction details: `Product Type`, `SKU`, `Quantity`, `Unit Price`, `Total Price`, `Purchase Date`, `Order Status`, `Payment Method`, `Shipping Type`
  - Ratings and add-ons: `Rating`, `Add-ons Purchased`, `Add-on Total`

---

## Project Steps

### 1. Data Loading & Preprocessing
- Load the CSV using `pandas`.
- Handle missing values and convert columns to appropriate types (`Quantity`, `Total Price`, `Purchase Date`).
- Extract new features:
  - `Purchase_Month` and `Purchase_DayOfWeek` from `Purchase Date`.
  - `Sale_Amount = Quantity × Unit Price`.
  - `Addons_Count` from the `Add-ons Purchased` column.
- Filter only `Completed` orders for meaningful purchase data.

### 2. Descriptive Analysis
- Display dataset info and missing values.
- Compute basic statistics for numerical columns.
- Value counts for categorical columns.
- Visualize:
  - Distribution of `Quantity` and `Total Price`.
  - Correlation heatmap for numerical features.
  - Boxplots of `Total Price` by categorical variables (excluding add-ons, grouping dates by month).

### 3. Clustering
- Preprocess features:
  - Encode categorical features with `OneHotEncoder`.
  - Scale numerical features with `StandardScaler`.
- Apply **KMeans clustering** on prepared features.
- Determine optimal number of clusters using **Silhouette Score**.
- Assign cluster labels to each transaction.
- Visualize clusters in 2D using **PCA** or **t-SNE**.
- Optionally, add **density contours or convex hulls** to illustrate cluster volumes.

### 4. Cluster Analysis
- Compute summary statistics per cluster:
  - Average `Quantity`, `Sale_Amount`, `Age`, `Rating`
  - Most common `Loyalty Member` status, `Payment Method`, and `Product Type`
- Use PCA scatterplots to visualize clusters with quantity information:
  - Color points by actual `Quantity` (continuous) or discretized levels (`Low`, `Medium`, `High`).

### 5. Cluster Assignment for New Transactions
- Use the trained `KMeans` and `ColumnTransformer` to assign new transactions to clusters.
- Optional: train a **supervised classifier** (`RandomForestClassifier`) to predict cluster labels from transaction features.
- Output cluster number and characteristics for any new input.

---

## Installation

Python 3.9+ is recommended. Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn kmodes
