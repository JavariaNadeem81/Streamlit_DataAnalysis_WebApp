<<<<<<< HEAD
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
dataset_options = ["None", "iris", "tips", "penguins"]
choose_dataset = st.selectbox("Choose a built-in dataset", dataset_options)

# 3. File Uploader
upload_file = st.file_uploader("Or upload your own file", type=["csv", "xlsx", "txt"])

# Data Loading Logic
df = None
if upload_file is not None:
    if upload_file.name.endswith(".csv") or upload_file.name.endswith(".txt"):
        df = pd.read_csv(upload_file)
    elif upload_file.name.endswith(".xlsx"):
        df = pd.read_excel(upload_file)
    st.success("Custom file loaded successfully!")
elif choose_dataset != "None":
    df = sns.load_dataset(choose_dataset)
    st.info(f"Loaded {choose_dataset} dataset")

# --- MAIN APP LOGIC ---
if df is not None:
    st.write("### Data Preview", df.head())

    # --- STEP A: DATA HEALTH CHECK & CLEANING ---
    st.divider()
    st.header("🔍 Data Health Overview")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Missing Values (Nulls):**")
        st.write(df.isnull().sum())
    
    with col2:
        st.write("**Duplicates:**")
        dupes = df.duplicated().sum()
        st.metric("Duplicate Rows", dupes)

    # Function to convert dataframe for download
    def convert_df(df_to_download):
        return df_to_download.to_csv(index=False).encode('utf-8')

    st.markdown("### 📥 Data Cleaning & Export")
    if st.button("Auto-Clean Data (Remove Nulls & Duplicates)"):
        df = df.drop_duplicates().dropna()
        st.success("Data Cleaned! (Refreshed Preview below)")
        st.write(df.head())
        
    csv_data = convert_df(df)
    st.download_button(
        label="Download Cleaned CSV",
        data=csv_data,
        file_name='cleaned_data.csv',
        mime='text/csv',
    )

    # --- STEP B: ML TYPE RECOMMENDATION ---
    st.divider()
    st.header("🤖 ML Model Suggester")
    
    target = st.selectbox("Select your Target Column (Label)", df.columns)
    
    if target:
        unique_vals = df[target].nunique()
        # Suggest Classification if target is text or has few unique categories
        if df[target].dtype == 'object' or unique_vals < 15:
            st.success("**Recommended Type: CLASSIFICATION**")
            st.write("Suggested Models: Random Forest, SVC, or Logistic Regression.")
        else:
            st.success("**Recommended Type: REGRESSION**")
            st.write("Suggested Models: Linear Regression, XGBoost, or Ridge Regression.")

    # --- STEP C: SCALING & NORMALIZATION ADVICE ---
    st.divider()
    st.subheader("⚖️ Scaling & Normalization Advice")
    
    # Select only numeric columns to avoid comparison errors with strings
    numerical_df = df.select_dtypes(include=[np.number])
    
    if not numerical_df.empty:
        max_val = numerical_df.max().max()
        min_val = numerical_df.min().min()
        
        if max_val > 100 or min_val < -100:
            st.warning(f"⚠️ **Normalization Required:** Large value range detected (Max: {max_val}). Use **StandardScaler** or **MinMaxScaler**.")
        else:
            st.success("✅ **Range OK:** Values are within a standard range.")
    else:
        st.info("No numerical columns found for scaling check.")

    # --- STEP D: MODEL PERFORMANCE GUIDE ---
    st.divider()
    st.header("📉 Model Performance Diagnostics")
    st.write("Enter your model scores below to check for Overfitting or Underfitting:")
    
    perf_col1, perf_col2 = st.columns(2)
    train_score = perf_col1.number_input("Enter Training Accuracy (0-1.0):", 0.0, 1.0, 0.85)
    test_score = perf_col2.number_input("Enter Testing Accuracy (0-1.0):", 0.0, 1.0, 0.70)

    if st.button("Analyze Fit"):
        diff = train_score - test_score
        st.markdown("### 🛠️ Diagnostic Results & Recommended Fixes")
        
        if diff > 0.15:
            st.error(f"❌ **OVERFITTING DETECTED:** Gap is {diff:.2f}. Model is not generalizing well.")
            st.info("💡 **How to fix Overfitting:**")
            st.write("""
                1. **Regularization:** Apply L1 (Lasso) or L2 (Ridge) penalties.
                2. **Reduce Complexity:** Decrease tree depth or remove unnecessary features.
                3. **Data Augmentation:** Gather more training samples.
                4. **Cross-Validation:** Use K-Fold to ensure stable performance.
            """)
        elif train_score < 0.50:
            st.warning(f"⚠️ **UNDERFITTING DETECTED:** Training score ({train_score:.2f}) is too low.")
            st.info("💡 **How to fix Underfitting:**")
            st.write("""
                1. **Increase Complexity:** Add more layers, neurons, or features.
                2. **Feature Engineering:** Create new meaningful variables from existing data.
                3. **Extended Training:** Increase epochs or training time.
                4. **Switch Models:** Move from linear models to non-linear ones (e.g., Random Forest).
            """)
        else:
            st.balloons()
            st.success("✅ **HEALTHY FIT:** Your model generalizes well. Ready for deployment!")

else:
=======
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
dataset_options = ["None", "iris", "tips", "penguins"]
choose_dataset = st.selectbox("Choose a built-in dataset", dataset_options)

# 3. File Uploader
upload_file = st.file_uploader("Or upload your own file", type=["csv", "xlsx", "txt"])

# Data Loading Logic
df = None
if upload_file is not None:
    if upload_file.name.endswith(".csv") or upload_file.name.endswith(".txt"):
        df = pd.read_csv(upload_file)
    elif upload_file.name.endswith(".xlsx"):
        df = pd.read_excel(upload_file)
    st.success("Custom file loaded successfully!")
elif choose_dataset != "None":
    df = sns.load_dataset(choose_dataset)
    st.info(f"Loaded {choose_dataset} dataset")

# --- MAIN APP LOGIC ---
if df is not None:
    st.write("### Data Preview", df.head())

    # --- STEP A: DATA HEALTH CHECK & CLEANING ---
    st.divider()
    st.header("🔍 Data Health Overview")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Missing Values (Nulls):**")
        st.write(df.isnull().sum())
    
    with col2:
        st.write("**Duplicates:**")
        dupes = df.duplicated().sum()
        st.metric("Duplicate Rows", dupes)

    # Function to convert dataframe for download
    def convert_df(df_to_download):
        return df_to_download.to_csv(index=False).encode('utf-8')

    st.markdown("### 📥 Data Cleaning & Export")
    if st.button("Auto-Clean Data (Remove Nulls & Duplicates)"):
        df = df.drop_duplicates().dropna()
        st.success("Data Cleaned! (Refreshed Preview below)")
        st.write(df.head())
        
    csv_data = convert_df(df)
    st.download_button(
        label="Download Cleaned CSV",
        data=csv_data,
        file_name='cleaned_data.csv',
        mime='text/csv',
    )

    # --- STEP B: ML TYPE RECOMMENDATION ---
    st.divider()
    st.header("🤖 ML Model Suggester")
    
    target = st.selectbox("Select your Target Column (Label)", df.columns)
    
    if target:
        unique_vals = df[target].nunique()
        # Suggest Classification if target is text or has few unique categories
        if df[target].dtype == 'object' or unique_vals < 15:
            st.success("**Recommended Type: CLASSIFICATION**")
            st.write("Suggested Models: Random Forest, SVC, or Logistic Regression.")
        else:
            st.success("**Recommended Type: REGRESSION**")
            st.write("Suggested Models: Linear Regression, XGBoost, or Ridge Regression.")

    # --- STEP C: SCALING & NORMALIZATION ADVICE ---
    st.divider()
    st.subheader("⚖️ Scaling & Normalization Advice")
    
    # Select only numeric columns to avoid comparison errors with strings
    numerical_df = df.select_dtypes(include=[np.number])
    
    if not numerical_df.empty:
        max_val = numerical_df.max().max()
        min_val = numerical_df.min().min()
        
        if max_val > 100 or min_val < -100:
            st.warning(f"⚠️ **Normalization Required:** Large value range detected (Max: {max_val}). Use **StandardScaler** or **MinMaxScaler**.")
        else:
            st.success("✅ **Range OK:** Values are within a standard range.")
    else:
        st.info("No numerical columns found for scaling check.")

    # --- STEP D: MODEL PERFORMANCE GUIDE ---
    st.divider()
    st.header("📉 Model Performance Diagnostics")
    st.write("Enter your model scores below to check for Overfitting or Underfitting:")
    
    perf_col1, perf_col2 = st.columns(2)
    train_score = perf_col1.number_input("Enter Training Accuracy (0-1.0):", 0.0, 1.0, 0.85)
    test_score = perf_col2.number_input("Enter Testing Accuracy (0-1.0):", 0.0, 1.0, 0.70)

    if st.button("Analyze Fit"):
        diff = train_score - test_score
        st.markdown("### 🛠️ Diagnostic Results & Recommended Fixes")
        
        if diff > 0.15:
            st.error(f"❌ **OVERFITTING DETECTED:** Gap is {diff:.2f}. Model is not generalizing well.")
            st.info("💡 **How to fix Overfitting:**")
            st.write("""
                1. **Regularization:** Apply L1 (Lasso) or L2 (Ridge) penalties.
                2. **Reduce Complexity:** Decrease tree depth or remove unnecessary features.
                3. **Data Augmentation:** Gather more training samples.
                4. **Cross-Validation:** Use K-Fold to ensure stable performance.
            """)
        elif train_score < 0.50:
            st.warning(f"⚠️ **UNDERFITTING DETECTED:** Training score ({train_score:.2f}) is too low.")
            st.info("💡 **How to fix Underfitting:**")
            st.write("""
                1. **Increase Complexity:** Add more layers, neurons, or features.
                2. **Feature Engineering:** Create new meaningful variables from existing data.
                3. **Extended Training:** Increase epochs or training time.
                4. **Switch Models:** Move from linear models to non-linear ones (e.g., Random Forest).
            """)
        else:
            st.balloons()
            st.success("✅ **HEALTHY FIT:** Your model generalizes well. Ready for deployment!")

else:
>>>>>>> 8de859ab018362e923c12eaa3403a8aaeb3eed33
    st.warning("Please upload a file or select a built-in dataset to begin analysis.")