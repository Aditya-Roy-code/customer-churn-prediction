import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: #0b0f19;
    }

    /* Main content */
    .main .block-container {
        padding-top: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
        max-width: 1500px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #263244;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: white;
    }

    /* Header */
    .hero {
        padding: 30px 35px;
        border-radius: 20px;
        margin-bottom: 30px;
        background: linear-gradient(
            135deg,
            #18243a 0%,
            #111827 55%,
            #172554 100%
        );
        border: 1px solid #263244;
        box-shadow: 0 10px 35px rgba(0,0,0,0.25);
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 17px;
        color: #9ca3af;
    }

    /* KPI cards */
    .metric-card {
        background: linear-gradient(
            145deg,
            #151c2b,
            #101622
        );
        border: 1px solid #263244;
        border-radius: 18px;
        padding: 22px;
        min-height: 130px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.20);
    }

    .metric-icon {
        font-size: 26px;
        margin-bottom: 10px;
    }

    .metric-title {
        color: #9ca3af;
        font-size: 14px;
        font-weight: 600;
    }

    .metric-value {
        color: #ffffff;
        font-size: 30px;
        font-weight: 800;
        margin-top: 5px;
    }

    /* Section headings */
    .section-title {
        color: white;
        font-size: 25px;
        font-weight: 750;
        margin-top: 30px;
        margin-bottom: 15px;
    }

    /* Chart containers */
    .chart-box {
        background: #111827;
        border: 1px solid #263244;
        border-radius: 18px;
        padding: 15px;
    }

    /* Prediction box */
    .prediction-box {
        background: linear-gradient(
            145deg,
            #151c2b,
            #0f172a
        );
        border: 1px solid #263244;
        border-radius: 20px;
        padding: 30px;
        margin-top: 20px;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        border: none;
        padding: 12px 20px;
        font-weight: 700;
        font-size: 16px;
        background: linear-gradient(
            90deg,
            #2563eb,
            #7c3aed
        );
        color: white;
    }

    .stButton > button:hover {
        background: linear-gradient(
            90deg,
            #3b82f6,
            #8b5cf6
        );
        color: white;
    }

    /* Dataframe */
    .stDataFrame {
        border-radius: 15px;
        overflow: hidden;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #6b7280;
        padding: 30px;
        font-size: 13px;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("models/churn_model.pkl")


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv("data/customer_churn.csv")


model = load_model()
df = load_data()

df.columns = df.columns.str.strip()


# =========================================================
# CLEAN CHURN FOR DASHBOARD
# =========================================================

dashboard_df = df.copy()

if dashboard_df["Churn"].dtype == "object":

    dashboard_df["Churn"] = (
        dashboard_df["Churn"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "yes": 1,
            "no": 0,
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0
        })
    )

dashboard_df["Churn"] = pd.to_numeric(
    dashboard_df["Churn"],
    errors="coerce"
)

dashboard_df["Years"] = pd.to_numeric(
    dashboard_df["Years"],
    errors="coerce"
)

dashboard_df["Total_Purchase"] = pd.to_numeric(
    dashboard_df["Total_Purchase"],
    errors="coerce"
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <h1 style="font-size:25px;">
        📊 Churn Analytics
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='color:#9ca3af;'>Customer Intelligence Platform</p>",
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "📋 Data Explorer",
            "📈 Churn Analysis",
            "🔮 Predict Churn"
        ]
    )

    st.divider()

    st.markdown(
        """
        <p style="color:#6b7280;font-size:12px;">
        Built with Python • Pandas • Scikit-learn • Streamlit
        </p>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# HERO HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">
            📊 Customer Churn Analytics
        </div>
        <div class="hero-subtitle">
            Analyze customer behavior, discover churn patterns,
            and predict customer retention risk using Machine Learning.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="section-title">📈 Business Overview</div>',
        unsafe_allow_html=True
    )

    total_customers = len(df)

    churned_customers = int(
        dashboard_df["Churn"].fillna(0).sum()
    )

    churn_rate = (
        churned_customers /
        total_customers
    ) * 100

    avg_years = dashboard_df["Years"].mean()

    avg_purchase = dashboard_df["Total_Purchase"].mean()


    # =====================================================
    # KPI CARDS
    # =====================================================

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">👥</div>
                <div class="metric-title">TOTAL CUSTOMERS</div>
                <div class="metric-value">{total_customers:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">⚠️</div>
                <div class="metric-title">CHURNED CUSTOMERS</div>
                <div class="metric-value">{churned_customers:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">📉</div>
                <div class="metric-title">CHURN RATE</div>
                <div class="metric-value">{churn_rate:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">⏳</div>
                <div class="metric-title">AVG. CUSTOMER YEARS</div>
                <div class="metric-value">{avg_years:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown(
        '<div class="section-title">💰 Customer Value</div>',
        unsafe_allow_html=True
    )

    st.info(
        f"Average customer purchase value: "
        f"**{avg_purchase:,.2f}**"
    )


    # =====================================================
    # QUICK DATA
    # =====================================================

    st.markdown(
        '<div class="section-title">👥 Recent Customer Data</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        df.head(8),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# DATA EXPLORER
# =========================================================

elif page == "📋 Data Explorer":

    st.markdown(
        '<div class="section-title">📋 Explore Customer Dataset</div>',
        unsafe_allow_html=True
    )

    search = st.text_input(
        "🔎 Search customer/company/location"
    )

    filtered_df = df.copy()

    if search:

        mask = filtered_df.astype(str).apply(
            lambda row: row.str.contains(
                search,
                case=False,
                na=False
            ).any(),
            axis=1
        )

        filtered_df = filtered_df[mask]


    st.write(
        f"Showing **{len(filtered_df)}** customers"
    )

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )


    st.markdown(
        '<div class="section-title">📌 Missing Values</div>',
        unsafe_allow_html=True
    )

    missing = df.isnull().sum()

    missing = missing[
        missing > 0
    ].sort_values(
        ascending=False
    )

    if len(missing) > 0:

        st.dataframe(
            missing.rename("Missing Values"),
            use_container_width=True
        )

    else:

        st.success(
            "✅ No missing values found!"
        )


# =========================================================
# CHURN ANALYSIS
# =========================================================

elif page == "📈 Churn Analysis":

    st.markdown(
        '<div class="section-title">📈 Churn Insights</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)


    # =====================================================
    # CHURN DISTRIBUTION
    # =====================================================

    with col1:

        st.markdown(
            "### 🎯 Churn Distribution"
        )

        churn_counts = (
            dashboard_df["Churn"]
            .value_counts()
            .sort_index()
        )

        fig, ax = plt.subplots(
            figsize=(7, 4)
        )

        ax.bar(
            ["Stayed", "Churned"],
            [
                churn_counts.get(0, 0),
                churn_counts.get(1, 0)
            ]
        )

        ax.set_ylabel("Customers")

        ax.set_facecolor("#111827")
        fig.patch.set_facecolor("#111827")

        ax.tick_params(
            colors="white"
        )

        ax.yaxis.label.set_color(
            "white"
        )

        ax.spines["bottom"].set_color(
            "#374151"
        )

        ax.spines["left"].set_color(
            "#374151"
        )

        st.pyplot(
            fig,
            use_container_width=True
        )


    # =====================================================
    # CHURN BY YEARS
    # =====================================================

    with col2:

        st.markdown(
            "### ⏳ Churn vs Customer Years"
        )

        fig, ax = plt.subplots(
            figsize=(7, 4)
        )

        sns.boxplot(
            data=dashboard_df,
            x="Churn",
            y="Years",
            ax=ax
        )

        ax.set_xticklabels(
            ["Stayed", "Churned"]
        )

        ax.set_xlabel("")
        ax.set_ylabel("Years")

        ax.set_facecolor("#111827")
        fig.patch.set_facecolor("#111827")

        ax.tick_params(
            colors="white"
        )

        st.pyplot(
            fig,
            use_container_width=True
        )


    # =====================================================
    # PURCHASE VS CHURN
    # =====================================================

    col3, col4 = st.columns(2)


    with col3:

        st.markdown(
            "### 💰 Purchase vs Churn"
        )

        fig, ax = plt.subplots(
            figsize=(7, 4)
        )

        sns.boxplot(
            data=dashboard_df,
            x="Churn",
            y="Total_Purchase",
            ax=ax
        )

        ax.set_xticklabels(
            ["Stayed", "Churned"]
        )

        ax.set_xlabel("")
        ax.set_ylabel("Total Purchase")

        ax.set_facecolor("#111827")
        fig.patch.set_facecolor("#111827")

        st.pyplot(
            fig,
            use_container_width=True
        )


    # =====================================================
    # NUM SITES
    # =====================================================

    with col4:

        st.markdown(
            "### 🏢 Sites vs Churn"
        )

        fig, ax = plt.subplots(
            figsize=(7, 4)
        )

        sns.boxplot(
            data=dashboard_df,
            x="Churn",
            y="Num_Sites",
            ax=ax
        )

        ax.set_xticklabels(
            ["Stayed", "Churned"]
        )

        ax.set_xlabel("")
        ax.set_ylabel("Number of Sites")

        ax.set_facecolor("#111827")
        fig.patch.set_facecolor("#111827")

        st.pyplot(
            fig,
            use_container_width=True
        )


    # =====================================================
    # LOCATION
    # =====================================================

    st.markdown(
        "### 📍 Churn Rate by Location"
    )

    location_churn = (
        dashboard_df
        .groupby("Location")["Churn"]
        .mean()
        .sort_values(
            ascending=False
        )
        .head(15)
    )

    st.bar_chart(
        location_churn
    )


# =========================================================
# PREDICT CHURN
# =========================================================

elif page == "🔮 Predict Churn":

    st.markdown(
        '<div class="section-title">🔮 Customer Churn Prediction</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter customer information to estimate "
        "their churn probability."
    )


    st.markdown(
        '<div class="prediction-box">',
        unsafe_allow_html=True
    )


    col1, col2 = st.columns(2)


    # =====================================================
    # LEFT SIDE
    # =====================================================

    with col1:

        st.markdown(
            "### 👤 Customer Information"
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35
        )

        total_purchase = st.number_input(
            "Total Purchase",
            min_value=0.0,
            value=10000.0
        )

        account_manager = st.selectbox(
            "Account Manager",
            [0, 1],
            format_func=lambda x:
                "Yes" if x == 1 else "No"
        )

        years = st.number_input(
            "Years with Company",
            min_value=0.0,
            max_value=100.0,
            value=3.0
        )


    # =====================================================
    # RIGHT SIDE
    # =====================================================

    with col2:

        st.markdown(
            "### 🏢 Company Information"
        )

        num_sites = st.number_input(
            "Number of Sites",
            min_value=1,
            max_value=100,
            value=3
        )


        locations = sorted(
            df["Location"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        location = st.selectbox(
            "Location",
            locations
        )


        companies = sorted(
            df["Company"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        company = st.selectbox(
            "Company",
            companies
        )


        onboard_year = st.number_input(
            "Onboard Year",
            min_value=2000,
            max_value=2030,
            value=2026
        )


        onboard_month = st.selectbox(
            "Onboard Month",
            list(range(1, 13)),
            index=0
        )


    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


    st.write("")


    # =====================================================
    # PREDICT
    # =====================================================

    if st.button(
        "🚀 ANALYZE CUSTOMER RISK"
    ):

        customer_data = pd.DataFrame({

            "Age": [age],

            "Total_Purchase": [
                total_purchase
            ],

            "Account_Manager": [
                account_manager
            ],

            "Years": [
                years
            ],

            "Num_Sites": [
                num_sites
            ],

            "Location": [
                location
            ],

            "Company": [
                company
            ],

            "Onboard_year": [
                onboard_year
            ],

            "Onboard_month": [
                onboard_month
            ]
        })


        prediction = model.predict(
            customer_data
        )[0]


        probability = model.predict_proba(
            customer_data
        )[0][1]


        st.divider()


        # =================================================
        # RESULT
        # =================================================

        if prediction == 1:

            st.error(
                "⚠️ HIGH CHURN RISK"
            )

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-icon">⚠️</div>
                    <div class="metric-title">
                        CHURN PROBABILITY
                    </div>
                    <div class="metric-value">
                        {probability * 100:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.warning(
                "This customer shows a high probability "
                "of leaving the company."
            )

        else:

            st.success(
                "✅ LOW CHURN RISK"
            )

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-icon">✅</div>
                    <div class="metric-title">
                        CHURN PROBABILITY
                    </div>
                    <div class="metric-value">
                        {probability * 100:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.info(
                "This customer is currently predicted "
                "to be likely to stay."
            )


        st.write("")

        st.write(
            "### Risk Meter"
        )

        st.progress(
            float(probability)
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        Customer Churn Analytics Dashboard •
        Machine Learning Project •
        Python + Scikit-learn + Streamlit
    </div>
    """,
    unsafe_allow_html=True
)