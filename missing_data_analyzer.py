import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


class MissingDataAnalyser:
    def __init__(self,df):
        self.df=df

    def show_missing_values(self):
        self.null_df = pd.DataFrame({
            "Column": self.df.columns,
            "Null Percentage ": self.df.isnull().mean()*100
        })

        st.subheader("Null Value Summary")
        st.dataframe(self.null_df)

    def classify_column(self, cat_threshold=10):
        self.col_info={'binary':[],'categorical':[],'continous':[],'datetime':[],'id':[],'constant':[]}
        for i in self.df.columns:
            if self.df[i].nunique(dropna=True)<=1:
                self.col_info['constant'].append(i)
                continue
            
            if pd.api.types.is_bool_dtype(self.df[i]):
                self.col_info['binary'].append(i)
                continue

            if pd.api.types.is_datetime64_any_dtype(self.df[i]):
                self.col_info['datetime'].append(i)
                continue

            if pd.api.types.is_numeric_dtype(self.df[i]):
                uni = self.df[i].nunique(dropna=True)
                if uni<=1:
                    self.col_info['constant'].append(i)
                elif uni<= cat_threshold:
                    self.col_info['categorical'].append(i)
                else:
                    self.col_info['continous'].append(i)
            else:
                uni = self.df[i].nunique(dropna=True)
                if uni==len(self.df):     
                    self.col_info['id'].append(i)
                else:
                    self.col_info['categorical'].append(i)
        return self.col_info



    