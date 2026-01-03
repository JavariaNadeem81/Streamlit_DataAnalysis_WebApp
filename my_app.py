import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Smart Data Analyzer", layout="wide")

# 1. Title and Header
st.title("🚀 Smart Data Analysis & ML Guide")
st.subheader("Analyze your data and get machine learning suggestions")

# 2. Dataset Selection Sidebar
st.sidebar.header("Data Settings")
dataset_options = ["None", "iris", "tips", "penguins"]
choose_dataset = st.sidebar.selectbox("Choose a built-in dataset", dataset_options)

# 3. File Uploader
upload_file = st.file_uploader("Or upload your own file (CSV, XLSX, TXT)", type=["csv", "xlsx", "txt"])

df = None

# Data Loading Logic
if upload_file is not None:
    if upload_file.name.endswith(".csv"):
        df = pd.read_csv(upload_file)
    elif upload_file.name.endswith(".xlsx"):
        df = pd.read_excel(upload_file)
    st.success("Custom file loaded successfully!")
elif choose_dataset != "None":
    df = sns.load_dataset(choose_dataset)
    st.success(f"Built-in '{choose_dataset}' dataset loaded!")

if df is not None:
    st.write("### Data Preview", df.head())

    # --- STEP A: SMART DATA HEALTH & CLEANING ---
    st.divider()
    st.header("🔍 Smart Data Health Overview")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Missing Values (Nulls):**")
        null_counts = df.isnull().sum()
        st.write(null_counts[null_counts > 0] if null_counts.sum() > 0 else "No Nulls Found!")
        
    with col2:
        st.write("**Row Statistics:**")
        dupes = df.duplicated().sum()
        st.metric("Duplicate Rows Found", dupes)
        st.write(f"Total Rows: {df.shape[0]} | Total Columns: {df.shape[1]}")

    # --- NORMALIZATION ALERT ---
    numerical_cols = df.select_dtypes(include=[np.number])
    if not numerical_cols.empty:
        if numerical_cols.max().max() > 100 or numerical_cols.min().min() < -100:
            st.warning("⚠️ **Normalization Alert:** Your data values are not scaled. Using 'StandardScaler' or 'MinMaxScaler' is highly recommended.")

    # --- SMART CLEANING BUTTON ---
    st.markdown("### 📥 Smart Cleaning (3M Approach)")
    if st.button("Run Auto-Clean (Handle Nulls & Duplicates)"):
        # Handle Duplicates
        old_count = len(df)
        df = df.drop_duplicates()
        removed_dupes = old_count - len(df)
        
        # Smart Imputation (Mean/Median/Mode)
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['int64', 'float64']:
                    # Use Median for numbers to avoid outlier influence
                    df[col] = df[col].fillna(df[col].median())
                else:
                    # Use Mode for categorical/text columns
                    df[col] = df[col].fillna(df[col].mode()[0])
        
        st.toast("Data Cleaned Successfully!", icon="✅")
        st.success(f"✅ Cleaned! {removed_dupes} Duplicates removed. Nulls handled via Smart Imputation (3M).")
        st.write(df.head())

    # --- STEP B: ML SUGGESTER WITH "WHY?" ---
    st.divider()
    st.header("🤖 ML Model Suggester")
    target_col = st.selectbox("Select your Target Column (Label)", df.columns)
    
    if target_col:
        is_numeric = df[target_col].dtype in ['int64', 'float64']
        unique_vals = df[target_col].nunique()
        
        if is_numeric and unique_vals > 10:
            st.info(f"Recommended Type: **REGRESSION**")
            with st.expander("❓ Why Regression?"):
                st.write(f"**Reason:** The target column `{target_col}` is numeric and contains many unique values ({unique_vals}). Since you are likely predicting a continuous value, Regression is the best choice.")
            st.write("**Suggested Models:** Linear Regression, XGBoost, or Random Forest Regressor.")
        else:
            st.info(f"Recommended Type: **CLASSIFICATION**")
            with st.expander("❓ Why Classification?"):
                st.write(f"**Reason:** The target column `{target_col}` has a limited number of unique values ({unique_vals}). This suggests you are grouping data into distinct categories, which is a Classification task.")
            st.write("**Suggested Models:** Logistic Regression, Random Forest, or Support Vector Machine (SVM).")

    # --- STEP C: MODEL DIAGNOSTICS & RECOMMENDED FIXES ---
    st.divider()
    st.header("📉 Model Performance Diagnostics")
    c1, c2 = st.columns(2)
    train_score = c1.number_input("Enter Training Accuracy (0-1.0):", 0.0, 1.0, 0.85)
    test_score = c2.number_input("Enter Testing Accuracy (0-1.0):", 0.0, 1.0, 0.70)
    
    if st.button("Analyze Fit"):
        diff = train_score - test_score
        st.markdown("---")
        st.subheader("🛠️ Recommended Fix (Best Approach)")

        if diff > 0.15:
            st.warning("⚠️ **Overfitting Detected!**")
            st.info("💡 **How to Fix Overfitting?**")
            st.write("""
            1. **Regularization:** Apply L1 (Lasso) or L2 (Ridge) penalties.
            2. **Simplify Model:** Reduce the maximum depth of trees or number of neurons.
            3. **Data Augmentation:** Gather more training samples to help the model generalize.
            4. **Dropout:** If using Neural Networks, add Dropout layers.
            5. **Cross-Validation:** Use K-Fold cross-validation to verify stability.
            """)
        elif train_score < 0.60:
            st.error("⚠️ **Underfitting Detected!**")
            st.info("💡 **How to Fix Underfitting?**")
            st.write("""
            1. **Increase Complexity:** Add more features or layers to your model.
            2. **Feature Engineering:** Create better input variables from existing data.
            3. **Train Longer:** Increase the number of training iterations (epochs).
            4. **Change Model:** If using a linear model, try a non-linear one like Random Forest or SVM.
            """)
        else:
            st.balloons()
            st.success("✅ **Good Fit!**")
            st.write("✨ Your model is performing well! It is now ready for **Deployment**.")

    # Download Button
    st.divider()
    st.download_button(
        label="Download Cleaned CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='final_cleaned_data.csv',
        mime='text/csv',
    )
else:
    st.info("Please upload a file or choose a dataset to begin.")