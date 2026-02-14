import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="Medical Insurance Dashboard",
    layout="wide"
)


st.markdown("""
    <style>
    .main {
        background-color: #f8fbff;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #f3e8ff, #e0f2ff);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #6a0dad, #9b59b6);
    color: white;
}

[data-testid="stSidebar"] .css-1v0mbdj {
    color: white;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #6a0dad, #9b59b6);
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: bold;
    border: none;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #5b0cb0, #8e44ad);
    color: white;
}

/* Selectbox & Sliders */
.stSelectbox div, .stSlider {
    border-radius: 10px !important;
}

</style>
""", unsafe_allow_html=True)


st.sidebar.title("🏥 Medical Insurance")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Intro", "📊 View Tables",  "🤖 Predict", "🧠 Health Profile","📈 EDA"]
)
st.sidebar.markdown("---")

df=pd.read_csv("medical_eda.csv")

if menu == "🏠 Intro":

    #  Custom Styling 
    st.markdown("""
        <style>
        .main-title {
            font-size: 42px;
            font-weight: 800;
            color: #003366;
            text-align: center;
        }

        .subtitle {
            font-size: 20px;
            text-align: center;
            color: #4f4f4f;
            margin-bottom: 30px;
        }

        .card {
            background-color: #ffffff;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.05);
            margin-bottom: 25px;
        }

        .section-title {
            font-size: 24px;
            font-weight: 700;
            color: #00509e;
            margin-bottom: 15px;
        }
        </style>
    """, unsafe_allow_html=True)

    # ---------- Title ----------
    st.markdown('<div class="main-title">🏥 Medical Insurance Cost Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">An End-to-End Regression & Deployment Project</div>', unsafe_allow_html=True)

    # ---------- Project Overview ----------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📌 Project Overview</div>', unsafe_allow_html=True)

    st.write("""
    This project builds an end-to-end regression pipeline to predict individual 
    medical insurance costs based on:

    • Age  
    • Gender  
    • BMI  
    • Smoking Status  
    • Number of Dependents  
    • Region  

    The workflow includes data preprocessing, feature engineering, training multiple regression models,
    experiment tracking with MLflow, and deploying the best model using Streamlit.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- Business Use Cases ----------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">💼 Business Use Cases</div>', unsafe_allow_html=True)

    st.write("""
    • Assisting insurance companies in determining personalized premiums.  

    • Helping individuals compare medical insurance policies based on their profile.  

    • Supporting healthcare consultants in estimating potential out-of-pocket costs.  

    • Providing cost transparency and increasing financial awareness among policyholders.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- Approach ----------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚙️ Project Approach</div>', unsafe_allow_html=True)

    st.write("""
    ### 🔹 Step 1: Data Preprocessing
    - Load and inspect dataset  
    - Handle missing, inconsistent, and duplicate records  
    - Encode categorical variables  
    - Perform feature engineering (BMI categories, interaction terms)  

    ### 🔹 Step 2: Medical Insurance Cost Prediction
    - Perform Exploratory Data Analysis (EDA)  
    - Train 5+ regression models:
        - Linear Regression  
        - Random Forest  
        - XGBoost  
        - (and others)
    - Evaluate using RMSE, MAE, R²  
    - Track experiments using MLflow  
    - Register best model in MLflow Model Registry  

    ### 🔹 Step 3: Streamlit App Development
    - Display EDA visual insights  
    - Accept user health & demographic inputs  
    - Predict estimated insurance cost  
    - Optionally show confidence intervals or error margins  
    """)
    
elif menu == "📊 View Tables":

    # ---------- Styling ----------
    st.markdown("""
        <style>
        .section-title {
            font-size: 30px;
            font-weight: 800;
            color: #003366;
            margin-bottom: 10px;
        }

        .card {
            background-color: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0px 4px 20px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">📊 Dataset Overview</div>', unsafe_allow_html=True)

    # ---------- Metrics Row ----------
    # ---------- Metrics Row ----------
    col1, col2, col3, col4, col5 = st.columns(5)

    # Basic metrics
    avg_charges = round(df["charges"].mean(), 2)
    avg_age = round(df["age"].mean(), 1)
    avg_bmi = round(df["bmi"].mean(), 1)
    avg_children = round(df["children"].mean(), 1)

    # Smoker %
    if "smoker" in df.columns:
        smoker_percentage = round((df["smoker"].value_counts(normalize=True)[1]) * 100, 1)
    else:
        smoker_percentage = 0

    col1.metric("💰 Avg Charges", f"{avg_charges}")
    col2.metric("🎂 Avg Age", avg_age)
    col3.metric("⚖ Avg BMI", avg_bmi)
    col4.metric("👨‍👩‍👧 Avg Dependents", avg_children)
    col5.metric("🚬 % Smokers", f"{smoker_percentage}%")

    st.markdown("---")

    # ---------- Column Selector ----------
    st.markdown("""
    <style>
    .card {
        background-color: #f3e8ff;  /* light purple */
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 20px rgba(128, 0, 128, 0.15);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)


    st.subheader("🔎 Explore Dataset")

    selected_columns = st.multiselect(
        "Select columns to display",
        options=df.columns,
        default=df.columns
    )

    temp_df = df[selected_columns]

    # ---------- Search Bar ----------
    search = st.text_input("🔍 Search in dataset")

    if search:
        temp_df = temp_df[
            temp_df.astype(str)
            .apply(lambda row: row.str.contains(search, case=False))
            .any(axis=1)
        ]

    st.dataframe(temp_df, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


elif menu == "🤖 Predict":

    import joblib
    import numpy as np

    model = joblib.load("models/xgb_model.pkl")
    scaler = joblib.load("models/scaler.pkl")

    # ------------------ INPUTS FIRST ------------------
    st.title(" PREDICT YOUR INSURANCE COST")

   

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("🎂 Age", 18, 100, 30)
        bmi = st.slider("⚖ BMI", 10.0, 50.0, 25.0)
        children = st.slider("👨‍👩‍👧 Dependents", 0, 5, 0)

    with col2:
        gender = st.selectbox("👤 Gender", ["male", "female"])
        smoker = st.selectbox("🚬 Smoker", ["yes", "no"])
        region = st.selectbox("📍 Region",
                              ["northeast", "northwest", "southeast", "southwest"])

    # ------------------ THEN ENCODING ------------------

    sex = 1 if gender == "male" else 0
    smoker_val = 1 if smoker == "yes" else 0

    region_northwest = 1 if region == "northwest" else 0
    region_southeast = 1 if region == "southeast" else 0
    region_southwest = 1 if region == "southwest" else 0

    # ------------------ SCALE ONLY NUMERIC ------------------

    numeric_data = np.array([[age, bmi, children]])
    scaled_numeric = scaler.transform(numeric_data)

    final_input = np.array([[
        scaled_numeric[0][0],
        sex,
        scaled_numeric[0][1],
        scaled_numeric[0][2],
        smoker_val,
        region_northwest,
        region_southeast,
        region_southwest
    ]])

    # ------------------ PREDICT ------------------
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("💰 Predict Insurance Cost"):

        prediction = model.predict(final_input)[0]

        st.markdown(f"""
        <div style="
        background: linear-gradient(135deg, #6a0dad, #9b59b6);
        color: white;
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        box-shadow: 0 15px 40px rgba(106, 13, 173, 0.4);
        margin-top: 30px;
        ">
        💰 Estimated Insurance Cost <br><br>
        Rs {round(prediction, 2)}
        </div>
        """, unsafe_allow_html=True)

elif menu == "🧠 Health Profile":

    st.markdown("""
    <h1 style='text-align:center; color:#6a0dad; font-size:40px;'>
    🧠 Health Profile Analyzer
    </h1>
    <p style='text-align:center; color:gray;'>
    Check your age category and BMI classification
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("🎂 Enter Your Age", 10, 100, 25)

    with col2:
        bmi = st.slider("⚖ Enter Your BMI", 10.0, 50.0, 22.0)

    # -------- Age Classification --------
# -------- Age Classification --------
    if 18 <= age <= 30:
        age_category = "Young"
        age_color = "#4CAF50"      # Green
    elif 31 <= age <= 45:
        age_category = "Adult"
        age_color = "#2196F3"      # Blue
    elif 46 <= age <= 60:
        age_category = "Middle Aged"
        age_color = "#ff9800"      # Orange
    else:
        age_category = "Senior"
        age_color = "#f44336"      # Red


    # -------- BMI Classification --------
    if bmi < 18.5:
        bmi_category = "Underweight"
        bmi_color = "#03A9F4"
    elif bmi < 25:
        bmi_category = "Normal Weight"
        bmi_color = "#4CAF50"
    elif bmi < 30:
        bmi_category = "Overweight"
        bmi_color = "#ff9800"
    else:
        bmi_category = "Obese"
        bmi_color = "#f44336"

    st.markdown("## 📊 Your Health Classification")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown(f"""
        <div style="
        background:{age_color};
        color:white;
        padding:30px;
        border-radius:20px;
        text-align:center;
        font-size:22px;
        font-weight:bold;
        box-shadow:0 10px 30px rgba(0,0,0,0.2);
        ">
        🎂 Age Category <br><br>
        {age_category}
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div style="
        background:{bmi_color};
        color:white;
        padding:30px;
        border-radius:20px;
        text-align:center;
        font-size:22px;
        font-weight:bold;
        box-shadow:0 10px 30px rgba(0,0,0,0.2);
        ">
        ⚖ BMI Category <br><br>
        {bmi_category}
        </div>
        """, unsafe_allow_html=True)
    # -------- Smoker Input --------
    smoker = st.selectbox("🚬 Are you a smoker?", ["yes", "no"])

    # -------- Risk Logic --------
    conditions = 0

    if smoker == "yes":
        conditions += 1
    if age > 40:
        conditions += 1
    if bmi > 29.9:
        conditions += 1

    if conditions == 3:
        risk_category = "High Risk"
        risk_color = "#f44336"   # Red
    elif conditions == 2:
        risk_category = "Moderate Risk"
        risk_color = "#ff9800"   # Orange
    else:
        risk_category = "Low Risk"
        risk_color = "#4CAF50"   # Green

    st.markdown("## ⚠ Risk Assessment")

    st.markdown(f"""
    <div style="
    background:{risk_color};
    color:white;
    padding:30px;
    border-radius:20px;
    text-align:center;
    font-size:24px;
    font-weight:bold;
    box-shadow:0 10px 30px rgba(0,0,0,0.2);
    margin-top:20px;
    ">
    ⚠ Overall Risk Level <br><br>
    {risk_category}
    </div>
    """, unsafe_allow_html=True)

elif menu == "📈 EDA":

    st.markdown("""
    <h1 style='text-align:center; color:#6a0dad; font-size:42px;'>
    📊 Exploratory Data Analysis
    </h1>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📌 Univariate",
        "📊 Bivariate",
        "📈 Multivariate",
        "🚨 Outliers",
        "🔗 Correlation"
    ])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Distribution of Insurance Charges")

            fig1, ax1 = plt.subplots()
            ax1.hist(df["charges"], bins=30)
            ax1.set_xlabel("Charges")
            ax1.set_ylabel("Frequency")

            st.pyplot(fig1)

            st.info("💡 Charges appear right-skewed, indicating a small group of individuals with very high medical costs.")

        with col2:
            st.subheader("Age Distribution")

            fig2, ax2 = plt.subplots()
            ax2.hist(df["age"], bins=20)
            ax2.set_xlabel("Age")
            ax2.set_ylabel("Count")

            st.pyplot(fig2)

            st.info("💡 Age distribution appears fairly balanced across working-age groups.")

            st.markdown("---")

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Smokers vs Non-Smokers")

            smoker_counts = df["smoker"].value_counts()

            fig3, ax3 = plt.subplots()
            ax3.bar(smoker_counts.index.astype(str), smoker_counts.values)

            st.pyplot(fig3)

            st.info("💡 0:Non Smoker 1:Smoker Majority of policyholders are non-smokers.")

        with col4:
            st.subheader("Average BMI")

            avg_bmi = round(df["bmi"].mean(), 2)

            st.markdown(f"""
            <div style="
            background: linear-gradient(135deg, #6a0dad, #9b59b6);
            color: white;
            padding: 35px;
            border-radius: 20px;
            text-align: center;
            font-size: 28px;
            font-weight: bold;
            box-shadow: 0 10px 30px rgba(106, 13, 173, 0.4);
            ">
            ⚖ Average BMI <br><br>
            {avg_bmi}
            </div>
            """, unsafe_allow_html=True)

            st.info("💡 This indicates the dataset leans toward overweight classification.")

            st.markdown("---")

        st.subheader("Region-wise Policyholder Distribution")

        region_counts = df["region"].value_counts()

        fig5, ax5 = plt.subplots()
        ax5.bar(region_counts.index, region_counts.values)

        st.pyplot(fig5)

        st.info("💡 Some regions have higher concentration of policyholders, which may impact regional pricing strategies.")


    with tab2:

        st.markdown("""
        <h2 style='color:#6a0dad;'>📊 Bivariate Analysis</h2>
        <p style='color:gray;'>Understanding relationships between two variables</p>
        """, unsafe_allow_html=True)

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Charges vs Age")

            fig1, ax1 = plt.subplots()
            ax1.scatter(df["age"], df["charges"])
            ax1.set_xlabel("Age")
            ax1.set_ylabel("Charges")

            st.pyplot(fig1)

            st.info("💡 Insurance charges generally increase with age, indicating higher health risks over time.")


        with col2:
            st.subheader("Average Charges: Smokers vs Non-Smokers")

            smoker_avg = df.groupby("smoker")["charges"].mean()

            fig2, ax2 = plt.subplots()
            ax2.bar(smoker_avg.index.astype(str), smoker_avg.values)

            st.pyplot(fig2)

            st.info("💡 Smokers typically incur significantly higher insurance costs compared to non-smokers.")

            st.markdown("---")

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("BMI vs Charges")

            fig3, ax3 = plt.subplots()
            ax3.scatter(df["bmi"], df["charges"])
            ax3.set_xlabel("BMI")
            ax3.set_ylabel("Charges")

            st.pyplot(fig3)

            st.info("💡 Higher BMI tends to correlate with increased medical expenses.")


        with col4:
            st.subheader("Average Charges by Gender")

            gender_avg = df.groupby("sex")["charges"].mean()

            fig4, ax4 = plt.subplots()
            ax4.bar(gender_avg.index.astype(str), gender_avg.values)

            st.pyplot(fig4)

            st.info("💡 0:Men 1: Women Differences in charges between genders may reflect lifestyle or health pattern variations.")


        st.markdown("---")

        st.subheader("Number of Children vs Average Charges")

        children_avg = df.groupby("children")["charges"].mean()

        fig5, ax5 = plt.subplots()
        ax5.bar(children_avg.index, children_avg.values)

        st.pyplot(fig5)

        st.info("💡 Insurance charges may increase slightly with dependents, but the effect is generally smaller than smoking or age.")


    with tab3:

        st.markdown("""
        <h2 style='color:#6a0dad;'>📈 Multivariate Analysis</h2>
        <p style='color:gray;'>Exploring combined impact of multiple health and demographic factors</p>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # =====================================================
        # 1️⃣ Smoking + Age Impact
        # =====================================================

        st.subheader("1️⃣ Smoking Status Combined with Age")

        pivot1 = df.pivot_table(
            values="charges",
            index="age",
            columns="smoker",
            aggfunc="mean"
        )

        st.line_chart(pivot1)

        st.info("💡 0: non smoker 1: smoker Smokers consistently incur higher costs across all age groups, and the gap widens as age increases.")

        st.markdown("---")

        # =====================================================
        # 2️⃣ Gender + Region Impact (Smokers Only)
        # =====================================================

        st.subheader("Gender & Region Impact on Charges (Smokers Only)")

        # Adjust if smoker is encoded
        if df["smoker"].dtype != "object":
            smoker_df = df[df["smoker"] == 1]
        else:
            smoker_df = df[df["smoker"] == "yes"]

        # If sex is encoded
        if df["sex"].dtype != "object":
            smoker_df["sex"] = smoker_df["sex"].map({0: "Male", 1: "Female"})

        pivot2 = smoker_df.pivot_table(
            values="charges",
            index="region",
            columns="sex",
            aggfunc="mean"
        )

        st.bar_chart(pivot2)

        st.info("💡1:Female 0:Male Among smokers, region and gender both influence average medical costs.")


        # =====================================================
        # 3️⃣ Age + BMI + Smoking Effect
        # =====================================================

        st.subheader("3️⃣ Age, BMI & Smoking Combined Effect")

        df["BMI_Category"] = pd.cut(
            df["bmi"],
            bins=[0, 18.5, 25, 30, 100],
            labels=["Underweight", "Normal", "Overweight", "Obese"]
        )

        pivot3 = df.pivot_table(
            values="charges",
            index="BMI_Category",
            columns="smoker",
            aggfunc="mean"
        )

        st.bar_chart(pivot3)

        st.info("💡0: Non smoker 1:Smoker Obese smokers show significantly higher medical expenses compared to other groups.")

        st.markdown("---")

        # =====================================================
        # 4️⃣ Obese Smokers vs Non-Obese Non-Smokers
        # =====================================================

        st.subheader("Obese Smokers vs Non-Obese Non-Smokers")

        # Handle encoding safely
        if df["smoker"].dtype != "object":
            obese_smokers = df[(df["bmi"] > 30) & (df["smoker"] == 1)]
            healthy_non_smokers = df[(df["bmi"] <= 30) & (df["smoker"] == 0)]
        else:
            obese_smokers = df[(df["bmi"] > 30) & (df["smoker"] == "yes")]
            healthy_non_smokers = df[(df["bmi"] <= 30) & (df["smoker"] == "no")]

        avg_obese = obese_smokers["charges"].mean()
        avg_healthy = healthy_non_smokers["charges"].mean()

        comparison_df = pd.DataFrame({
            "Group": ["Obese Smokers", "Non-Obese Non-Smokers"],
            "Average Charges": [avg_obese, avg_healthy]
        })

        st.bar_chart(comparison_df.set_index("Group"))

        difference = round(avg_obese - avg_healthy, 2)

        st.success(f"💰 Obese smokers pay approximately {difference} more on average.")


    with tab4:

        st.markdown("""
        <h2 style='color:#6a0dad;'>🚨 Outlier Detection</h2>
        <p style='color:gray;'>Identifying extreme values that may influence model performance</p>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # =====================================================
        # 1️⃣ Outliers in Charges
        # =====================================================

        st.subheader("1️⃣ Outliers in Medical Insurance Charges")

        fig1, ax1 = plt.subplots()
        ax1.boxplot(df["charges"], vert=False)
        ax1.set_xlabel("Charges")

        st.pyplot(fig1)

        st.info("💡 The boxplot highlights extreme high-cost individuals that may heavily influence regression models.")

        # Show top 5 highest charges
        st.subheader("Top 5 Individuals with Highest Charges")

        top5 = df.sort_values(by="charges", ascending=False).head(5)

        st.dataframe(top5[["age", "sex", "bmi", "children", "smoker", "region", "charges"]])

        st.markdown("---")

        # =====================================================
        # 2️⃣ Extreme BMI Values
        # =====================================================

        st.subheader("2️⃣ Extreme BMI Values")

        fig2, ax2 = plt.subplots()
        ax2.boxplot(df["bmi"], vert=False)
        ax2.set_xlabel("BMI")

        st.pyplot(fig2)

        st.info("💡 Extreme BMI values may skew model predictions and indicate higher health risk categories.")

        # Show extreme BMI rows
        extreme_bmi = df[(df["bmi"] < 15) | (df["bmi"] > 45)]

        st.subheader("Individuals with Extreme BMI")

        if len(extreme_bmi) > 0:
            st.dataframe(extreme_bmi[["age", "sex", "bmi", "smoker", "charges"]])
        else:
            st.success("No extreme BMI values detected beyond defined thresholds.")

    with tab5:

        st.markdown("""
        <h2 style='color:#6a0dad;'>🔗 Correlation Analysis</h2>
        <p style='color:gray;'>Understanding relationships between numeric variables</p>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # =====================================================
        # 1️⃣ Correlation Matrix
        # =====================================================

        st.subheader("1️⃣ Correlation Matrix (Numeric Features)")

        numeric_df = df.select_dtypes(include=["int64", "float64"])

        corr_matrix = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(8,6))
        cax = ax.matshow(corr_matrix)
        fig.colorbar(cax)

        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=90)
        ax.set_yticklabels(corr_matrix.columns)

        st.pyplot(fig)

        st.info("💡 Stronger positive or negative values indicate stronger linear relationships between variables.")

        st.markdown("---")

        # =====================================================
        # 2️⃣ Correlation with Target (Charges)
        # =====================================================

        st.subheader("2️⃣ Correlation with Target Variable (Charges)")

        charges_corr = corr_matrix["charges"].sort_values(ascending=False)

        st.bar_chart(charges_corr)

        st.markdown("### 📌 Correlation Values")
        st.dataframe(charges_corr)

        # Highlight strongest predictor
        strongest_feature = charges_corr.index[1]  # exclude charges itself

        st.success(f"🎯 Feature with strongest correlation to charges: **{strongest_feature}**")













        

        






