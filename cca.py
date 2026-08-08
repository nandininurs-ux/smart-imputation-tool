import streamlit as st
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

class CompleteCaseAnalysis():
    def __init__(self, df):
        self.df = df.copy()
        self.new_df1 = self.df.copy()

    def display(self):
        self.check_null()
        self.dropp()
        self.num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.cols = self.df.columns.tolist()
        self.valid_num_cols = [col for col in self.num_cols if col in self.cols]
        
        st.write('#### Numerical Columns (Preview)')
        st.dataframe(self.df[self.valid_num_cols].head())
        
        st.write('### New dataset with null values removed (CCA)')
        st.dataframe(self.new_df1.head())
        
        st.write('### Categorical Columns')
        st.dataframe(self.df[self.cat_cols].head())
       
    def check_null(self):
        self.cols = [i for i in self.df.columns if 0.00 < self.df[i].isnull().mean() < 0.05]
        if len(self.cols) == 0:
            st.write('### The given dataset has no columns suitable for CCA (< 5% missing)')

    def dropp(self):
        self.cols = [i for i in self.df.columns if 0.00 < self.df[i].isnull().mean() < 0.05]
        rows = self.df[self.cols].dropna().index
        self.new_df1 = self.df.loc[rows].copy()
        return self.new_df1

    def comp(self):
        self.dropp()
        for i in self.num_cols:
            fig, ax = plt.subplots()
            self.df[i].hist(bins=50, ax=ax, density=True, color='red', label='Original')
            self.new_df1[i].hist(bins=50, ax=ax, color='green', density=True, alpha=0.8, label='CCA')
            plt.title(f"Distribution Comparison: {i}")
            plt.legend()
            st.pyplot(fig)
            
            st.write('#### Is the distribution of data overlapping cleanly?')
            self.observation = st.text_input(f"Observation for {i}", key=f"obs_{i}")
            if self.observation:
                st.write("You entered:", self.observation)

    def categ(self):
        self.cat_cols = self.new_df1.select_dtypes(include=['object','category']).columns
        for i in self.cat_cols:
            self.temp = pd.concat(
                [self.df[i].value_counts() / len(self.df), 
                 self.new_df1[i].value_counts() / len(self.new_df1)], 
                axis=1
            )
            self.temp.columns = ['Original %', 'CCA %']
            st.write(f"#### Category Distribution: {i}")
            st.dataframe(self.temp)

class SimpleImpute():
    def __init__(self, df):
        self.df = df.copy()
        st.write('### Advanced Imputation Preview')
        self.split()

    def split(self):
        self.y_cols = [col for col in self.df.columns if self.df[col].dtype == 'bool' or self.df[col].dropna().isin([0, 1, '0', '1']).all()]
        
        if not self.y_cols:
            st.warning("⚠️ No binary target column (Y) detected. ML scoring algorithms may be skipped, but imputation will continue.")
            self.y_train, self.y_test = None, None
            self.x = self.df.copy()
            self.x_train, self.x_test = train_test_split(self.x, test_size=0.2, random_state=2)
        else:
            st.write(f"**Target Column(s) Detected:** {self.y_cols}")
            self.y = self.df[self.y_cols]
            self.x = self.df.drop(columns=self.y_cols)
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                self.x, self.y, test_size=0.2, random_state=2
            )
            
        st.write(f"Data Split: Train ({self.x_train.shape[0]} rows) | Test ({self.x_test.shape[0]} rows)")

    def mmimp(self):
        if not hasattr(self, 'x_train'):
            st.error("Data must be split first.")
            return
            
        cols_with_missing = [col for col in self.x_train.columns if self.x_train[col].isnull().sum() > 0]
        self.mean_ = {}
        self.median_ = {}
        
        if len(cols_with_missing) == 0:
            st.info("No missing values found in the features to impute.")
            return

        for i in cols_with_missing:
            if pd.api.types.is_numeric_dtype(self.x_train[i]):
                self.mean_[i] = self.x_train[i].mean()
                self.median_[i] = self.x_train[i].median()
                
                self.x_train[f'{i}_mean'] = self.x_train[i].fillna(self.mean_[i])
                self.x_train[f'{i}_median'] = self.x_train[i].fillna(self.median_[i])
                self.x_test[f'{i}_mean'] = self.x_test[i].fillna(self.mean_[i])
                self.x_test[f'{i}_median'] = self.x_test[i].fillna(self.median_[i])
        
        st.success("✅ Mean & Median Imputation Applied!")
        st.dataframe(self.x_train.head())

    def knn_impute(self):
        if not hasattr(self, 'x_train'):
            return
            
        st.write("### 🧠 K-Nearest Neighbors (KNN) Imputation")
        k_val = st.slider("Select number of neighbors (k)", min_value=1, max_value=15, value=5)
        weight_type = st.radio("Select weight type:", ['uniform', 'distance'])
        
        num_cols = self.x_train.select_dtypes(include=['number']).columns.tolist()
        
        if self.x_train[num_cols].isnull().sum().sum() > 0:
            knn = KNNImputer(n_neighbors=k_val, weights=weight_type)
            knn.fit(self.x_train[num_cols])
            
            self.x_train_knn = self.x_train.copy()
            self.x_train_knn[num_cols] = knn.transform(self.x_train[num_cols])
            
            st.success(f"✅ KNN Imputation applied successfully! (k={k_val})")
            st.dataframe(self.x_train_knn.head())
        else:
            st.info("No missing values found in numerical columns for KNN.")

    def auto_impute(self):
        if not hasattr(self, 'y_train') or self.y_train is None:
            st.error("Auto-Imputer requires a valid binary target (Y) column to score algorithms.")
            return
            
        st.write("### 🤖 Automatic Value Imputer (GridSearchCV)")
        num_cols = self.x_train.select_dtypes(include=['number']).columns.tolist()
        cat_cols = self.x_train.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not num_cols:
            st.warning("No numerical columns found to optimize.")
            return

        num_pipeline = Pipeline(steps=[('imputer', SimpleImputer())])
        cat_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent'))])
        
        preprocessor = ColumnTransformer(transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ], remainder='passthrough')
        
        clf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000))
        ])
        
        param_grid = {'preprocessor__num__imputer__strategy': ['mean', 'median']}
        
        with st.spinner("Testing algorithms..."):
            try:
                grid_search = GridSearchCV(clf, param_grid, cv=5)
                grid_search.fit(self.x_train, self.y_train.iloc[:, 0].values.ravel())
                
                best_strategy = grid_search.best_params_['preprocessor__num__imputer__strategy']
                st.success(f"✅ Best Numerical Strategy Found: **{best_strategy.upper()}**")
                st.write(f"Model Accuracy with this strategy: **{grid_search.best_score_ * 100:.2f}%**")
                
            except Exception as e:
                st.error(f"Auto-Imputer encountered an error. Details: {e}")