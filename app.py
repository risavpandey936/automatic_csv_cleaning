import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Auto CSV Cleaner & EDA", layout="wide")

st.title("📊 Automatic CSV Cleaning & EDA Pipeline")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])


def remove_id_columns(df):
    """
    Automatically removes ID-like columns
    """
    id_keywords = ["id", "user_id", "customer_id", "index", "unnamed"]

    cols_to_drop = []

    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in id_keywords):
            cols_to_drop.append(col)

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        return df, cols_to_drop
    return df, []


if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("🔍 Original Dataset Preview")
        st.dataframe(df.head())

        # ---------------- Remove ID Columns ----------------
        df, removed_cols = remove_id_columns(df)
        if removed_cols:
            st.info(f"Automatically removed ID-like columns: {removed_cols}")

        # ---------------- Basic Info ----------------
        st.subheader("📐 Dataset Information")
        col1, col2 = st.columns(2)

        with col1:
            st.write("Shape:", df.shape)
        with col2:
            st.write("Total Columns:", len(df.columns))

        st.write("Column Names:")
        st.write(list(df.columns))

        # ---------------- Missing Values ----------------
        st.subheader("❌ Missing Values Analysis")
        missing_values = df.isnull().sum()
        st.write(missing_values)

        # ---------------- Cleaning ----------------
        st.subheader("🧹 Data Cleaning")

        for col in df.columns:
            if df[col].dtype == "object":
                if not df[col].mode().empty:
                    df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].fillna(df[col].mean())

        # Convert all numeric to float
        numeric_cols = df.select_dtypes(include=np.number).columns
        df[numeric_cols] = df[numeric_cols].astype(float)

        st.success("Dataset cleaned successfully!")

        # ---------------- Column Types ----------------
        categorical_cols = df.select_dtypes(include="object").columns
        numerical_cols = df.select_dtypes(include=np.number).columns

        st.subheader("🔠 Categorical Columns")
        st.write("Total:", len(categorical_cols))
        st.write(list(categorical_cols))

        for col in categorical_cols:
            st.write(f"Unique values in '{col}': {df[col].nunique()}")
            st.write(df[col].value_counts())

        st.subheader("🔢 Numerical Columns")
        st.write("Total:", len(numerical_cols))
        st.write(list(numerical_cols))

        # ---------------- Summary Statistics ----------------
        st.subheader("📊 Summary Statistics")
        st.dataframe(df.describe())

        # ---------------- Skewness ----------------
        st.subheader("📉 Skewness Detection")
        skewness = df[numerical_cols].skew()
        st.write(skewness)

        # ---------------- Outlier Detection ----------------
        st.subheader("🚨 Outlier Detection (IQR Method)")
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            st.write(f"{col}: {len(outliers)} outliers detected")

        # ================= EDA ==================
        st.subheader("📊 Exploratory Data Analysis")

        # -------- Numerical Analysis --------
        st.markdown("## 📈 Numerical Feature Analysis")
        for col in numerical_cols:
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))

            sns.histplot(df[col], kde=True, ax=ax[0])
            ax[0].set_title(f"Histogram of {col}")

            sns.boxplot(x=df[col], ax=ax[1])
            ax[1].set_title(f"Boxplot of {col}")

            st.pyplot(fig)

        # -------- Correlation Heatmap --------
        if len(numerical_cols) > 1:
            st.markdown("## 🔥 Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(df[numerical_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        # -------- Categorical Analysis --------
        st.markdown("## 📊 Categorical Feature Analysis")
        for col in categorical_cols:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(x=df[col], ax=ax)
            plt.xticks(rotation=45)
            ax.set_title(f"Countplot of {col}")
            st.pyplot(fig)

        # ---------------- Download ----------------
        st.subheader("⬇ Download Cleaned Dataset")
        cleaned_csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Cleaned CSV",
            cleaned_csv,
            "cleaned_dataset.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")

