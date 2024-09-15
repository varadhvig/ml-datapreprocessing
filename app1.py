# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import plotly.express as px
import missingno as msno
import matplotlib.pyplot as plt
import io

def load_preloaded_data():
    # Synthetic Finance Dataset
    np.random.seed(0)
    data = {
        'Age': np.random.randint(18, 70, size=1000),
        'Income': np.random.normal(50000, 15000, size=1000).astype(int),
        'Gender': np.random.choice(['Male', 'Female', 'Other'], size=1000),
        'Purchased': np.random.choice(['Yes', 'No'], size=1000),
        'Credit_Score': np.random.randint(300, 850, size=1000)
    }
    df = pd.DataFrame(data)
    # Introduce missing values
    for col in df.columns:
        df.loc[df.sample(frac=0.1).index, col] = np.nan
    return df

def main():
    st.set_page_config(page_title="ML Data Cleaning Playground", layout="wide")
    st.title("📊 ML Data Cleaning Interactive Playground")
    st.markdown("""
    Welcome to the ML Data Cleaning Interactive Playground! This tool allows you to perform essential data preprocessing steps interactively.
    Choose between using a preloaded finance dataset or upload your own dataset (CSV/Parquet) to get started.
    """)

    # Dataset selection
    dataset_option = st.sidebar.radio("Select Dataset Option:", ("Use Preloaded Finance Dataset", "Upload Your Dataset"))

    if dataset_option == "Use Preloaded Finance Dataset":
        df = load_preloaded_data()
        st.success("Preloaded Finance Dataset Loaded Successfully!")
    elif dataset_option == "Upload Your Dataset":
        uploaded_file = st.file_uploader("Upload your dataset (CSV or Parquet)", type=['csv', 'parquet'])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_parquet(uploaded_file)
                st.success("Dataset uploaded successfully!")
            except Exception as e:
                st.error(f"Error loading file: {e}")
                st.stop()
        else:
            st.info("Awaiting for dataset to be uploaded.")
            st.stop()

    st.subheader("📋 Data Overview")
    st.write(f"**Dataset Shape:** {df.shape[0]} rows, {df.shape[1]} columns")

    # Display data types
    st.write("**Data Types:**")
    st.write(df.dtypes)

    # Display summary statistics
    st.write("**Summary Statistics:**")
    st.write(df.describe(include='all').transpose())

    # Display data preview
    st.write("**Dataset Preview:**")
    st.dataframe(df.head())

    st.sidebar.header("1. Handle Missing Values")

    missing_option = st.sidebar.selectbox("Choose how to handle missing values:", 
                                          ("Do Nothing", "Drop Rows", "Drop Columns", "Impute"))

    if missing_option == "Drop Rows":
        df = df.dropna()
        st.success("Dropped all rows with missing values.")
    elif missing_option == "Drop Columns":
        df = df.dropna(axis=1)
        st.success("Dropped all columns with missing values.")
    elif missing_option == "Impute":
        impute_method = st.sidebar.selectbox("Imputation Method:", ("Mean", "Median", "Mode"))
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                if impute_method == "Mean":
                    df[col].fillna(df[col].mean(), inplace=True)
                elif impute_method == "Median":
                    df[col].fillna(df[col].median(), inplace=True)
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        st.success(f"Imputed missing values using {impute_method}.")

    st.subheader("Dataset after Handling Missing Values")
    st.dataframe(df.head())

    if st.checkbox("Show Missing Data Heatmap"):
        st.subheader("Missing Data Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        msno.heatmap(df, ax=ax)
        st.pyplot(fig)

    st.sidebar.header("2. Encode Categorical Variables")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if categorical_cols:
        encoding_option = st.sidebar.selectbox("Encoding Method:", ("None", "Label Encoding", "One-Hot Encoding"))
        
        if encoding_option == "Label Encoding":
            le = LabelEncoder()
            for col in categorical_cols:
                df[col] = le.fit_transform(df[col].astype(str))
            st.success("Applied Label Encoding to categorical variables.")
        
        elif encoding_option == "One-Hot Encoding":
            df = pd.get_dummies(df, columns=categorical_cols)
            st.success("Applied One-Hot Encoding to categorical variables.")
    else:
        st.sidebar.write("No categorical variables to encode.")

    st.subheader("Dataset after Encoding")
    st.dataframe(df.head())

    st.sidebar.header("3. Feature Scaling")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    scaling_option = st.sidebar.selectbox("Scaling Method:", ("None", "Standard Scaling", "Min-Max Scaling"))

    if scaling_option == "Standard Scaling":
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        st.success("Applied Standard Scaling to numeric features.")

    elif scaling_option == "Min-Max Scaling":
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        st.success("Applied Min-Max Scaling to numeric features.")

    st.subheader("Dataset after Feature Scaling")
    st.dataframe(df.head())

    st.sidebar.header("4. Remove Outliers")
    outlier_option = st.sidebar.checkbox("Remove Outliers Using IQR")

    if outlier_option:
        Q1 = df[numeric_cols].quantile(0.25)
        Q3 = df[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        before_rows = df.shape[0]
        df = df[~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
        after_rows = df.shape[0]
        st.success(f"Removed outliers. Rows before: {before_rows}, after: {after_rows}")

        # Visualize box plots after outlier removal
        if st.checkbox("Show Box Plots After Outlier Removal"):
            st.subheader("Box Plots After Outlier Removal")
            fig = px.box(df[numeric_cols])
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dataset after Outlier Removal")
    st.dataframe(df.head())

    st.sidebar.header("5. Data Visualization")

    # Histogram
    if st.sidebar.checkbox("Show Histograms"):
        st.subheader("Histograms")
        for col in numeric_cols:
            fig = px.histogram(df, x=col, nbins=30, title=f'Histogram of {col}')
            st.plotly_chart(fig, use_container_width=True)

    # Scatter Plot
    if st.sidebar.checkbox("Show Scatter Plot"):
        st.subheader("Scatter Plot")
        x_axis = st.sidebar.selectbox("X-Axis", options=numeric_cols, key='scatter_x')
        y_axis = st.sidebar.selectbox("Y-Axis", options=[col for col in numeric_cols if col != x_axis], key='scatter_y')
        color_option = st.sidebar.selectbox("Color By (Optional)", options=[None] + categorical_cols, key='scatter_color')
        if color_option:
            fig = px.scatter(df, x=x_axis, y=y_axis, color=color_option, title=f'Scatter Plot of {y_axis} vs {x_axis}')
        else:
            fig = px.scatter(df, x=x_axis, y=y_axis, title=f'Scatter Plot of {y_axis} vs {x_axis}')
        st.plotly_chart(fig, use_container_width=True)

    # Correlation Matrix
    if st.sidebar.checkbox("Show Correlation Matrix"):
        st.subheader("Correlation Matrix")
        corr = df.corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu')
        st.plotly_chart(fig, use_container_width=True)

    # PCA Visualization
    if st.sidebar.checkbox("Show PCA"):
        st.subheader("PCA Plot")
        if len(numeric_cols) < 2:
            st.error("Need at least two numeric columns for PCA.")
        else:
            pca_components = st.sidebar.slider("Number of PCA Components", min_value=2, max_value=min(len(numeric_cols), 10), value=2)
            pca = PCA(n_components=pca_components)
            principal_components = pca.fit_transform(df[numeric_cols].dropna())
            pc_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(pca_components)])
            target = None
            if 'Purchased_Yes' in df.columns:
                target = 'Purchased_Yes'
            elif 'Purchased_No' in df.columns:
                target = 'Purchased_No'
            if target and target in df.columns:
                fig = px.scatter(pc_df, x='PC1', y='PC2', color=df[target].astype(str), title='PCA Scatter Plot')
            else:
                fig = px.scatter(pc_df, x='PC1', y='PC2', title='PCA Scatter Plot')
            st.plotly_chart(fig, use_container_width=True)

    st.sidebar.header("6. Download Cleaned Data")
    if st.sidebar.button("Download Cleaned CSV"):
        to_download = io.BytesIO()
        df.to_csv(to_download, index=False)
        to_download.seek(0)
        st.sidebar.download_button(
            label="📥 Download Cleaned Data as CSV",
            data=to_download,
            file_name='cleaned_data.csv',
            mime='text/csv',
        )

    if st.sidebar.button("Download Cleaned Parquet"):
        to_download = io.BytesIO()
        df.to_parquet(to_download, index=False)
        to_download.seek(0)
        st.sidebar.download_button(
            label="📥 Download Cleaned Data as Parquet",
            data=to_download,
            file_name='cleaned_data.parquet',
            mime='application/octet-stream',
        )

    st.subheader("📝 Final Cleaned Dataset Preview")
    st.dataframe(df.head())
    st.write(f"**Processed Dataset Shape:** {df.shape[0]} rows, {df.shape[1]} columns")

    st.markdown("### 📈 Data Summary")
    st.write(df.describe(include='all').transpose())

if __name__ == "__main__":
    main()

