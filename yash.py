import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gender_guesser.detector as gender
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from datetime import datetime
import os

# ---------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------
st.set_page_config(
    page_title="Enhanced Customer Insights Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------
# CUSTOM STYLING
# ---------------------------------------------
st.markdown("""
<style>
.main-header {font-size:3rem; font-weight:bold; color:#1f77b4; text-align:center;}
.metric-card {background:linear-gradient(135deg,#36D1DC 0%,#5B86E5 100%);
padding:1.2rem;border-radius:10px;color:white;text-align:center;}
[data-testid="stMetricValue"]{color:#fff;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------
# DATA LOADERS
# ---------------------------------------------
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file:
        df = pd.read_csv(uploaded_file, parse_dates=['Last_Purchase_Date'], dayfirst=True)
        return df
    return None

@st.cache_data
def process_data(df):
    # Clean names
    d = gender.Detector()
    df['First_Name'] = df['Name'].str.split(' ').str[0]
    df['Gender'] = df['First_Name'].apply(lambda x: d.get_gender(x))
    df['Gender'] = df['Gender'].replace({'male':'Male','mostly_male':'Male','female':'Female','mostly_female':'Female'}).fillna('Unknown')

    # Purchase segments
    df['Purchase_Segment'] = pd.cut(df['Purchase_Count'], [0, 5, 12, float('inf')], labels=['Low','Medium','High'])

    # Sentiment from feedback text (if available)
    if 'Feedback_Text' in df.columns:
        df['Sentiment_Polarity'] = df['Feedback_Text'].astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)
        df['Sentiment_Category'] = pd.cut(df['Sentiment_Polarity'], [-1, -0.1, 0.1, 1], labels=['Negative','Neutral','Positive'])

    # RFM Segmentation
    df['Recency'] = (datetime.now() - df['Last_Purchase_Date']).dt.days
    df['Frequency'] = df['Purchase_Count']
    df['Monetary'] = df['Total_Spent']
    df['RFM_Score'] = pd.qcut(df['Recency'], 3, labels=[3,2,1]).astype(int) + \
                      pd.qcut(df['Frequency'], 3, labels=[1,2,3]).astype(int) + \
                      pd.qcut(df['Monetary'], 3, labels=[1,2,3]).astype(int)
    return df

# ---------------------------------------------
# LOAD USER DATA
# ---------------------------------------------
st.markdown('<h1 class="main-header">📊 Enhanced Customer Insights Analyzer</h1>', unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Upload Customer CSV", type=["csv"])
df = load_data(uploaded_file)

if df is None:
    st.warning("Please upload a dataset to continue.")
    st.stop()

df = process_data(df)
st.success("✅ Dataset processed successfully!")

# ---------------------------------------------
# FILTERS
# ---------------------------------------------
gender_filter = st.sidebar.multiselect("Gender", df['Gender'].unique(), default=df['Gender'].unique())
segment_filter = st.sidebar.multiselect("Purchase Segment", ['Low','Medium','High'], default=['Low','Medium','High'])
state_filter = st.sidebar.multiselect("State", sorted(df['State'].unique()), default=df['State'].unique())
df_filtered = df[(df['Gender'].isin(gender_filter)) & (df['Purchase_Segment'].isin(segment_filter)) & (df['State'].isin(state_filter))]

# ---------------------------------------------
# KPIs
# ---------------------------------------------
col1,col2,col3,col4,col5 = st.columns(5)
col1.metric("Total Customers", f"{len(df_filtered):,}")
col2.metric("Avg Feedback Score", f"{df_filtered['Feedback_Score'].mean():.2f}")
col3.metric("Total Purchases", f"{df_filtered['Purchase_Count'].sum():,}")
col4.metric("Avg Purchase Frequency", f"{df_filtered['Purchase_Count'].mean():.1f}")
col5.metric("Coverage States", f"{df_filtered['State'].nunique()}")

# ---------------------------------------------
# SENTIMENT INSIGHTS TAB
# ---------------------------------------------
tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(["Overview","Demographics","Purchases","Feedback","Geography","Predictive Insights"])

with tab4:
    if 'Feedback_Text' in df_filtered.columns:
        st.subheader("Sentiment Distribution")
        sentiment_counts = df_filtered['Sentiment_Category'].value_counts()
        fig_sent = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig_sent, use_container_width=True)

with tab6:
    st.subheader("🔮 Churn Prediction (Sample Model)")
    churn_df = df_filtered.copy()
    churn_df = churn_df.dropna(subset=['Feedback_Score','Purchase_Count'])
    
    X = churn_df[['Feedback_Score','Purchase_Count','Total_Spent']]
    y = np.where(churn_df['Purchase_Segment']=='Low',1,0)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X,y)
    churn_df['Churn_Prob'] = model.predict_proba(X)[:,1]
    
    st.write("Sample Prediction Results")
    st.dataframe(churn_df[['Name','Churn_Prob']].head(15))
    st.plotly_chart(px.histogram(churn_df, x='Churn_Prob', nbins=20, title="Predicted Churn Probability Distribution"), use_container_width=True)

st.divider()
st.markdown("<center>📈 Built with Streamlit, Plotly & AI | 2025 Edition</center>", unsafe_allow_html=True)
