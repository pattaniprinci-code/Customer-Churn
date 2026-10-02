import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Churn ML Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "Artifacts")

RAW_FILE = os.path.join(ARTIFACTS_DIR, "raw.csv")
CLEAN_FILE = os.path.join(ARTIFACTS_DIR, "clean_data.csv")
TRAIN_FILE = os.path.join(ARTIFACTS_DIR, "train.csv")
TEST_FILE = os.path.join(ARTIFACTS_DIR, "test.csv")

MODEL_FILE = os.path.join(
    ARTIFACTS_DIR,
    "linear_regression.pkl"
)

SCALER_FILE = os.path.join(
    ARTIFACTS_DIR,
    "scaler.pkl"
)


# =========================================================
# LOAD DATA FUNCTION
# =========================================================

@st.cache_data
def load_csv(file_path):

    if os.path.exists(file_path):

        try:
            return pd.read_csv(file_path)

        except Exception as e:

            st.error(
                f"Unable to read file:\n{file_path}\n\n{e}"
            )

            return None

    return None


# =========================================================
# LOAD MODEL / PICKLE FUNCTION
# =========================================================

@st.cache_resource
def load_pickle(file_path):

    if not os.path.exists(file_path):

        return None

    # -----------------------------------------------------
    # First try normal pickle
    # -----------------------------------------------------

    try:

        with open(file_path, "rb") as file:

            return pickle.load(file)

    except Exception:

        pass


    # -----------------------------------------------------
    # Try joblib
    # -----------------------------------------------------

    try:

        import joblib

        return joblib.load(file_path)

    except Exception:

        return None


# =========================================================
# LOAD FILES
# =========================================================

raw_df = load_csv(RAW_FILE)

clean_df = load_csv(CLEAN_FILE)

train_df = load_csv(TRAIN_FILE)

test_df = load_csv(TEST_FILE)

model = load_pickle(MODEL_FILE)

scaler = load_pickle(SCALER_FILE)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📊 Customer Churn ML")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Dashboard",
        "📁 Raw Data",
        "🧹 Data Cleaning",
        "📊 EDA / Charts",
        "📈 Model Results",
        "🔮 Prediction",
        "📋 Train Data",
        "🧪 Test Data",
        "ℹ️ Project Information"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Customer Churn Prediction\n\n"
    "Machine Learning Project"
)


# =========================================================
# 1. DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    st.title("🏠 Customer Churn ML Dashboard")

    st.write(
        "Complete overview of the Customer Churn Machine Learning Project."
    )

    st.markdown("---")

    if raw_df is not None:

        total_rows = raw_df.shape[0]

        total_columns = raw_df.shape[1]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "👥 Total Customers",
            f"{total_rows:,}"
        )

        col2.metric(
            "📊 Total Columns",
            total_columns
        )

        if "churn" in raw_df.columns:

            churn_count = raw_df["churn"].astype(str).str.lower().isin(
                ["yes", "1", "true"]
            ).sum()

            no_churn = total_rows - churn_count

            col3.metric(
                "🔴 Churn",
                f"{churn_count:,}"
            )

            col4.metric(
                "🟢 No Churn",
                f"{no_churn:,}"
            )

            if total_rows > 0:

                churn_rate = (
                    churn_count / total_rows
                ) * 100

            else:

                churn_rate = 0

            st.markdown("---")

            st.subheader("📌 Churn Overview")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Churn Rate",
                f"{churn_rate:.2f}%"
            )

            c2.metric(
                "Churn Customers",
                f"{churn_count:,}"
            )

            c3.metric(
                "Non-Churn Customers",
                f"{no_churn:,}"
            )

        st.markdown("---")

        st.subheader("📋 Dataset Preview")

        st.dataframe(
            raw_df.head(10),
            use_container_width=True
        )

    else:

        st.error("raw.csv file not found!")


# =========================================================
# 2. RAW DATA
# =========================================================

elif page == "📁 Raw Data":

    st.title("📁 Raw Data")

    if raw_df is not None:

        st.success(
            "Raw dataset loaded successfully."
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Rows",
            f"{raw_df.shape[0]:,}"
        )

        col2.metric(
            "Columns",
            raw_df.shape[1]
        )

        col3.metric(
            "Missing Values",
            int(
                raw_df.isnull().sum().sum()
            )
        )

        st.markdown("---")

        search = st.text_input(
            "🔎 Search in data"
        )

        display_df = raw_df.copy()

        if search:

            mask = display_df.astype(str).apply(
                lambda row: row.str.contains(
                    search,
                    case=False,
                    na=False
                ).any(),
                axis=1
            )

            display_df = display_df[mask]

        st.subheader("📋 Dataset")

        st.dataframe(
            display_df,
            use_container_width=True,
            height=500
        )

        st.write(
            f"Showing {len(display_df):,} rows"
        )

        st.markdown("---")

        st.subheader("🔤 Data Types")

        dtype_df = pd.DataFrame({
            "Column": raw_df.columns,
            "Data Type": raw_df.dtypes.astype(str).values
        })

        st.dataframe(
            dtype_df,
            use_container_width=True
        )

        st.subheader("❓ Missing Values")

        missing_df = pd.DataFrame({
            "Column": raw_df.columns,
            "Missing Values": raw_df.isnull().sum().values
        })

        missing_df = missing_df.sort_values(
            "Missing Values",
            ascending=False
        )

        st.dataframe(
            missing_df,
            use_container_width=True
        )

    else:

        st.error("raw.csv file not found!")


# =========================================================
# 3. DATA CLEANING
# =========================================================

elif page == "🧹 Data Cleaning":

    st.title("🧹 Data Cleaning")

    if raw_df is not None:

        st.subheader("📌 Dataset Information")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Rows",
            f"{raw_df.shape[0]:,}"
        )

        col2.metric(
            "Total Columns",
            raw_df.shape[1]
        )

        col3.metric(
            "Duplicate Rows",
            int(
                raw_df.duplicated().sum()
            )
        )

        st.markdown("---")

        st.subheader("❓ Missing Value Analysis")

        missing = raw_df.isnull().sum()

        missing_df = pd.DataFrame({
            "Column": missing.index,
            "Missing Values": missing.values
        })

        missing_df = missing_df[
            missing_df["Missing Values"] > 0
        ]

        if len(missing_df) > 0:

            st.dataframe(
                missing_df,
                use_container_width=True
            )

        else:

            st.success(
                "No missing values found."
            )

        st.markdown("---")

        st.subheader("🔁 Duplicate Analysis")

        duplicate_count = raw_df.duplicated().sum()

        if duplicate_count == 0:

            st.success(
                "No duplicate rows found."
            )

        else:

            st.warning(
                f"{duplicate_count:,} duplicate rows found."
            )

        st.markdown("---")

        st.subheader("🔢 Numerical Columns")

        numerical_columns = raw_df.select_dtypes(
            include=np.number
        ).columns.tolist()

        st.write(numerical_columns)

        st.subheader("🔤 Categorical Columns")

        categorical_columns = raw_df.select_dtypes(
            exclude=np.number
        ).columns.tolist()

        st.write(categorical_columns)

        if clean_df is not None:

            st.markdown("---")

            st.subheader("✅ Cleaned Dataset")

            c1, c2 = st.columns(2)

            c1.metric(
                "Cleaned Rows",
                f"{clean_df.shape[0]:,}"
            )

            c2.metric(
                "Cleaned Columns",
                clean_df.shape[1]
            )

            st.dataframe(
                clean_df.head(20),
                use_container_width=True
            )

    else:

        st.error("raw.csv file not found!")


# =========================================================
# 4. EDA / CHARTS
# =========================================================

elif page == "📊 EDA / Charts":

    st.title("📊 Exploratory Data Analysis")

    if raw_df is not None:

        df = raw_df.copy()

        if "churn" in df.columns:

            st.subheader("🔴 Churn Distribution")

            churn_counts = df["churn"].value_counts()

            st.bar_chart(churn_counts)

            st.markdown("---")

        if (
            "gender" in df.columns
            and "churn" in df.columns
        ):

            st.subheader("👨‍👩‍👧 Gender vs Churn")

            gender_churn = pd.crosstab(
                df["gender"],
                df["churn"]
            )

            st.bar_chart(gender_churn)

            st.markdown("---")

        if (
            "contract" in df.columns
            and "churn" in df.columns
        ):

            st.subheader("📄 Contract vs Churn")

            contract_churn = pd.crosstab(
                df["contract"],
                df["churn"]
            )

            st.bar_chart(contract_churn)

            st.markdown("---")

        if (
            "payment_method" in df.columns
            and "churn" in df.columns
        ):

            st.subheader("💳 Payment Method vs Churn")

            payment_churn = pd.crosstab(
                df["payment_method"],
                df["churn"]
            )

            st.bar_chart(payment_churn)

            st.markdown("---")

        if "age" in df.columns:

            st.subheader("🎂 Age Distribution")

            st.bar_chart(
                df["age"].value_counts().sort_index()
            )

            st.markdown("---")

        if "tenure" in df.columns:

            st.subheader("📅 Tenure Distribution")

            tenure_data = (
                df["tenure"]
                .value_counts()
                .sort_index()
            )

            st.line_chart(
                tenure_data
            )

            st.markdown("---")

        if "monthly_charges" in df.columns:

            st.subheader("💰 Monthly Charges")

            st.line_chart(
                df["monthly_charges"]
                .head(100)
                .reset_index(drop=True)
            )

            st.markdown("---")

        if "total_charges" in df.columns:

            st.subheader("💵 Total Charges")

            st.line_chart(
                df["total_charges"]
                .head(100)
                .reset_index(drop=True)
            )

            st.markdown("---")

        st.subheader("🔗 Correlation Matrix")

        numerical_df = df.select_dtypes(
            include=np.number
        )

        if numerical_df.shape[1] >= 2:

            correlation = numerical_df.corr()

            st.dataframe(
                correlation,
                use_container_width=True
            )

        else:

            st.info(
                "Not enough numerical columns for correlation."
            )

    else:

        st.error("raw.csv file not found!")


# =========================================================
# 5. MODEL RESULTS
# =========================================================

elif page == "📈 Model Results":

    st.title("📈 Model Results")

    col1, col2 = st.columns(2)

    if model is not None:

        col1.success(
            "✅ Linear Regression model loaded."
        )

    else:

        col1.error(
            "❌ Linear Regression model could not be loaded."
        )

    if scaler is not None:

        col2.success(
            "✅ Scaler loaded."
        )

    else:

        col2.error(
            "❌ Scaler could not be loaded."
        )

    st.markdown("---")

    if model is None:

        st.warning(
            "The linear_regression.pkl file is invalid or corrupted."
        )

        st.info(
            "Please create the model file again from your training code."
        )

    if scaler is None:

        st.warning(
            "The scaler.pkl file is invalid or corrupted."
        )

    if train_df is not None:

        st.subheader("📋 Training Dataset")

        c1, c2 = st.columns(2)

        c1.metric(
            "Training Rows",
            f"{train_df.shape[0]:,}"
        )

        c2.metric(
            "Training Columns",
            train_df.shape[1]
        )

    if test_df is not None:

        st.subheader("🧪 Testing Dataset")

        c1, c2 = st.columns(2)

        c1.metric(
            "Testing Rows",
            f"{test_df.shape[0]:,}"
        )

        c2.metric(
            "Testing Columns",
            test_df.shape[1]
        )

    st.markdown("---")

    st.info(
        "Model: Linear Regression\n\n"
        "Target: Total Charges"
    )

    st.warning(
        "R² score shown here should be connected to the "
        "actual score saved/calculated during your model training."
    )


# =========================================================
# 6. PREDICTION
# =========================================================

elif page == "🔮 Prediction":

    st.title("🔮 Customer Total Charges Prediction")

    st.write(
        "Enter customer details to predict Total Charges."
    )

    if model is None:

        st.error(
            "❌ linear_regression.pkl file is invalid or corrupted."
        )

        st.info(
            "Please regenerate the Linear Regression model file "
            "from your model training code."
        )

    elif scaler is None:

        st.error(
            "❌ scaler.pkl file is invalid or corrupted."
        )

    else:

        col1, col2 = st.columns(2)

        with col1:

            customer_id = st.number_input(
                "Customer ID",
                min_value=0,
                value=1
            )

            age = st.number_input(
                "Age",
                min_value=1,
                max_value=120,
                value=30
            )

            gender = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

            tenure = st.number_input(
                "Tenure (Months)",
                min_value=0,
                value=12
            )

        with col2:

            monthly_charges = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                value=50.0
            )

            contract = st.selectbox(
                "Contract",
                [
                    "Month-to-month",
                    "One year",
                    "Two year"
                ]
            )

            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer",
                    "Credit card"
                ]
            )

            churn = st.selectbox(
                "Churn",
                ["No", "Yes"]
            )

        st.markdown("---")

        if st.button(
            "🔮 Predict Total Charges",
            use_container_width=True
        ):

            try:

                input_data = pd.DataFrame({

                    "customer_id": [customer_id],

                    "age": [age],

                    "gender": [gender],

                    "tenure": [tenure],

                    "monthly_charges": [
                        monthly_charges
                    ],

                    "contract": [contract],

                    "payment_method": [
                        payment_method
                    ],

                    "churn": [churn]

                })

                st.subheader("📋 Input Data")

                st.dataframe(
                    input_data,
                    use_container_width=True
                )

                # -------------------------------------------------
                # Convert categorical values
                # -------------------------------------------------

                input_encoded = pd.get_dummies(
                    input_data
                )

                # -------------------------------------------------
                # Match model feature columns
                # -------------------------------------------------

                if hasattr(
                    model,
                    "feature_names_in_"
                ):

                    model_columns = model.feature_names_in_

                    input_encoded = input_encoded.reindex(
                        columns=model_columns,
                        fill_value=0
                    )

                elif hasattr(
                    scaler,
                    "feature_names_in_"
                ):

                    scaler_columns = scaler.feature_names_in_

                    input_encoded = input_encoded.reindex(
                        columns=scaler_columns,
                        fill_value=0
                    )

                # -------------------------------------------------
                # Scale input
                # -------------------------------------------------

                input_scaled = scaler.transform(
                    input_encoded
                )

                # -------------------------------------------------
                # Prediction
                # -------------------------------------------------

                prediction = model.predict(
                    input_scaled
                )

                result = float(
                    prediction[0]
                )

                st.success(
                    f"💰 Predicted Total Charges: ₹{result:,.2f}"
                )

            except Exception as e:

                st.error(
                    "Prediction could not be completed."
                )

                st.exception(e)


# =========================================================
# 7. TRAIN DATA
# =========================================================

elif page == "📋 Train Data":

    st.title("📋 Training Data")

    if train_df is not None:

        col1, col2 = st.columns(2)

        col1.metric(
            "Rows",
            f"{train_df.shape[0]:,}"
        )

        col2.metric(
            "Columns",
            train_df.shape[1]
        )

        st.markdown("---")

        st.subheader("Training Dataset")

        st.dataframe(
            train_df,
            use_container_width=True,
            height=600
        )

    else:

        st.error(
            "train.csv file not found!"
        )


# =========================================================
# 8. TEST DATA
# =========================================================

elif page == "🧪 Test Data":

    st.title("🧪 Testing Data")

    if test_df is not None:

        col1, col2 = st.columns(2)

        col1.metric(
            "Rows",
            f"{test_df.shape[0]:,}"
        )

        col2.metric(
            "Columns",
            test_df.shape[1]
        )

        st.markdown("---")

        st.subheader("Testing Dataset")

        st.dataframe(
            test_df,
            use_container_width=True,
            height=600
        )

        st.markdown("---")

        st.subheader("📊 Numerical Summary")

        numerical_test = test_df.select_dtypes(
            include=np.number
        )

        if not numerical_test.empty:

            st.dataframe(
                numerical_test.describe(),
                use_container_width=True
            )

    else:

        st.error(
            "test.csv file not found!"
        )


# =========================================================
# 9. PROJECT INFORMATION
# =========================================================

elif page == "ℹ️ Project Information":

    st.title("ℹ️ Project Information")

    st.subheader("📌 Project Title")

    st.write(
        "Customer Churn Prediction using Machine Learning"
    )

    st.markdown("---")

    st.subheader("🎯 Problem Statement")

    st.write(
        "The objective of this project is to analyze customer "
        "data and use Machine Learning techniques to predict "
        "customer-related outcomes."
    )

    st.markdown("---")

    st.subheader("📊 Dataset Information")

    st.write(
        "The dataset contains customer information such as:"
    )

    st.write(
        """
        • Customer ID  
        • Age  
        • Gender  
        • Tenure  
        • Monthly Charges  
        • Contract  
        • Payment Method  
        • Churn  
        • Total Charges
        """
    )

    st.markdown("---")

    st.subheader("🤖 Machine Learning Algorithm")

    st.write(
        "Linear Regression"
    )

    st.markdown("---")

    st.subheader("🛠️ Technologies Used")

    st.write(
        """
        • Python  
        • Pandas  
        • NumPy  
        • Scikit-learn  
        • Matplotlib  
        • Streamlit  
        • Machine Learning
        """
    )

    st.markdown("---")

    st.subheader("🔄 Project Workflow")

    st.write(
        """
        1. Data Collection  
        2. Data Ingestion  
        3. Data Cleaning  
        4. Exploratory Data Analysis  
        5. Data Preprocessing  
        6. Feature Encoding  
        7. Feature Scaling  
        8. Model Training  
        9. Model Evaluation  
        10. Prediction  
        11. Streamlit Deployment
        """
    )

    st.markdown("---")

    st.success(
        "✅ Customer Churn Machine Learning Project"
    )