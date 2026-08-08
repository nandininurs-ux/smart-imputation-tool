import streamlit as st
import pandas as pd
from missing_data_analyzer import MissingDataAnalyser
from cca import CompleteCaseAnalysis, SimpleImpute

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Smart Missing Value Imputer",
    page_icon="🧹",
    layout="wide"
)

# -----------------------------------
# Title
# -----------------------------------
st.title("🧹 Smart Missing Value Imputer")
st.write("Upload a dataset to analyze missing values and recommend the best imputation strategy.")

# -----------------------------------
# Upload File
# -----------------------------------
uploaded_file = st.file_uploader("Upload CSV or Excel File", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.session_state["dataset"] = df
        st.success("✅ Dataset Loaded Successfully!")

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Dataset Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", int(df.isnull().sum().sum()))

        st.subheader("Column Data Types")
        dtype_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str)
        })
        st.dataframe(dtype_df)

        # Missing Value Analysis
        analyzer = MissingDataAnalyser(df)
        summary = analyzer.show_missing_values()

        # -----------------------------------
        # Choose Imputation Strategy 
        # -----------------------------------
        st.subheader("🛠️ Choose Your Data Cleaning Strategy")
        
        strategy = st.radio(
            "Select a method to handle the missing values:", 
            ["1. Complete Case Analysis (Drop Rows)", "2. Advanced Imputation (Fill Values)"]
        )

        if strategy == "1. Complete Case Analysis (Drop Rows)":
            cca = CompleteCaseAnalysis(df)
            cca.display()
            cca.comp()
            cca.categ()

        elif strategy == "2. Advanced Imputation (Fill Values)":
            # Using the unified name: SimpleImpute
            imputer = SimpleImpute(df)
            
            adv_method = st.selectbox(
                "Select which algorithm to use:", 
                ["Select...", "Mean/Median Imputer", "KNN Imputer", "Auto-Imputer (GridSearchCV)"]
            )
            
            if adv_method == "Mean/Median Imputer":
                imputer.mmimp()
            elif adv_method == "KNN Imputer":
                imputer.knn_impute()
            elif adv_method == "Auto-Imputer (GridSearchCV)":
                imputer.auto_impute()

    except Exception as e:
        st.error(f"Error loading dataset:\n\n{e}")
else:
    st.info("👆 Please upload a CSV or Excel file.")