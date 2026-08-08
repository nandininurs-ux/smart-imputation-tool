# smart-imputation-tool
An interactive web application built with Streamlit that automates missing data analysis and applies advanced machine learning imputation techniques (including K-NN and GridSearchCV) to clean datasets

#  Smart Missing Value Imputer

##  Overview
Data cleaning can take up to 80% of a data scientist's time, and handling missing values is one of the biggest bottlenecks. The **Smart Missing Value Imputer** is an interactive, end-to-end data preprocessing tool that automates the diagnosis and treatment of missing data. 

Users can upload any CSV or Excel dataset, instantly visualize the missingness profile, and apply both statistical and machine learning-based imputation pipelines without writing a single line of code.

##  Key Features

* **Automated Data Profiling:** Instantly scans uploaded datasets to generate comprehensive summaries of missing values, column data types, and structural dimensions.
* **Complete Case Analysis (CCA):** Automatically identifies columns with less than 5% missing data and allows users to safely drop rows while comparing pre- and post-drop distributions.
* **Univariate Imputation:** Injects Mean or Median values into numerical features while maintaining a transparent comparison against the original data.
* **Multivariate ML Imputation (K-NN):** Utilizes Scikit-Learn's `KNNImputer` to predict and fill missing values based on the Euclidean distance of nearest neighbor rows.
* **Auto-Imputer (GridSearchCV):** An intelligent pipeline that tests multiple imputation strategies (Mean vs. Median) against a baseline Logistic Regression model to automatically select and apply the strategy that yields the highest accuracy.

##  Tech Stack

* **Frontend/UI:** Streamlit
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning / Pipelines:** Scikit-Learn (`SimpleImputer`, `KNNImputer`, `Pipeline`, `ColumnTransformer`, `GridSearchCV`)
* **Data Visualization:** Matplotlib

##  How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YourUsername/smart-imputation-tool.git](https://github.com/YourUsername/smart-imputation-tool.git)
   cd smart-imputation-tool
