import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gender_guesser.detector as gender
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from datetime import datetime
import os
import warnings

warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="Enhanced Customer Insights Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size:3rem; font-weight:bold; color:#1f77b4; text-align:center; margin-bottom:1rem;
}
.metric-card {
    background:linear-gradient(135deg,#36D1DC 0%,#5B86E5 100%);
    padding:1.2rem; border-radius:10px; color:white; text-align:center;
}
[data-testid="stMetricValue"] {
    color:#fff;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 2rem;
}
</style>
""", unsafe_allow_html=True)

# Data loading & processing
@st.cache_data
def load_data(uploaded_file, default_path=None):
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, parse_dates=['Last_Purchase_Date'], dayfirst=True)
        elif default_path and os.path.exists(default_path):
            df = pd.read_csv(default_path, parse_dates=['Last_Purchase_Date'], dayfirst=True)
        else:
            return None
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def process_data(df):
    d = gender.Detector()
    df['Name'] = df['Name'].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
    df[['First_Name', 'Last_Name']] = df['Name'].str.split(' ', n=1, expand=True)
    df['Last_Name'] = df['Last_Name'].fillna('')
    
    df['Gender'] = df['First_Name'].apply(lambda x: d.get_gender(x))
    df['Gender'] = df['Gender'].replace({
        'male': 'Male', 'mostly_male': 'Male',
        'female': 'Female', 'mostly_female': 'Female',
        'andy': 'Unknown'
    })
    
    df['Purchase_Segment'] = pd.cut(
        df['Purchase_Count'],
        bins=[0,5,12,float('inf')],
        labels=['Low','Medium','High']
    )
    
    df['Last_Purchase_Date'] = pd.to_datetime(df['Last_Purchase_Date'], errors='coerce')
    df['Purchase_Month'] = df['Last_Purchase_Date'].dt.to_period('M').astype(str)
    df['Purchase_Year'] = df['Last_Purchase_Date'].dt.year

    if 'Feedback_Text' in df.columns:
        df['Sentiment_Polarity'] = df['Feedback_Text'].astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)
        df['Sentiment_Category'] = pd.cut(
            df['Sentiment_Polarity'],
            bins=[-1, -0.1, 0.1, 1],
            labels=['Negative', 'Neutral', 'Positive']
        )
    
    if 'Total_Spent' in df.columns:
        df['Total_Spent'] = pd.to_numeric(df['Total_Spent'], errors='coerce').fillna(0)
        df['Monetary'] = df['Total_Spent']
    else:
        df['Monetary'] = 0

    df['Recency'] = (datetime.now() - df['Last_Purchase_Date']).dt.days
    df['Frequency'] = df['Purchase_Count']
    df['Recency'] = df['Recency'].fillna(df['Recency'].max())

    df['R_Score'] = pd.qcut(df['Recency'], 3, labels=[3,2,1]).astype(int)
    df['F_Score'] = pd.qcut(df['Frequency'], 3, labels=[1,2,3]).astype(int)
    
    try:
        df['M_Score'] = pd.qcut(df['Monetary'], 3, labels=[1,2,3]).astype(int)
    except ValueError:
        df['M_Score'] = 1  

    df['RFM_Score'] = df['R_Score'] + df['F_Score'] + df['M_Score']
    return df

# Main app
st.markdown('<h1 class="main-header">📊 Enhanced Customer Insights Analyzer</h1>', unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("Upload Customer Dataset (CSV)", type=["csv"])
default_file = os.path.join(os.getcwd(), "2000_dataset.csv")
df = load_data(uploaded_file, default_file)

if df is None:
    st.error("❌ No dataset available! Please upload a CSV file.")
    st.stop()
else:
    st.success(f"✅ Dataset loaded with {len(df):,} records.")

df = process_data(df)

gender_filter = st.sidebar.multiselect("Select Gender", options=df['Gender'].unique(), default=list(df['Gender'].unique()))
segment_filter = st.sidebar.multiselect("Purchase Segment", options=['Low','Medium','High'], default=['Low','Medium','High'])
state_filter = st.sidebar.multiselect("Select States", options=sorted(df['State'].unique()), default=list(df['State'].unique()))

df_filtered = df[
    (df['Gender'].isin(gender_filter)) &
    (df['Purchase_Segment'].isin(segment_filter)) &
    (df['State'].isin(state_filter))
]

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Customers", f"{len(df_filtered):,}")
col2.metric("Avg Feedback Score", f"{df_filtered['Feedback_Score'].mean():.2f}")
col3.metric("Total Purchases", f"{df_filtered['Purchase_Count'].sum():,}")
col4.metric("Avg Purchase Frequency", f"{df_filtered['Purchase_Count'].mean():.1f}")
col5.metric("Covered States", f"{df_filtered['State'].nunique()}")

st.divider()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview", "Demographics", "Purchases", "Feedback", "Geography", "Predictive Insights"
])

with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df_filtered.head(100), use_container_width=True, height=400)
    st.subheader("Statistics")
    st.write(df_filtered.describe())
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button("Download Filtered Data", csv, f"customer_data_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

with tab2:
    st.subheader("Gender Distribution")
    gender_counts = df_filtered['Gender'].value_counts()
    fig_gender = px.pie(values=gender_counts.values, names=gender_counts.index, hole=0.4, color_discrete_sequence=px.colors.qualitative.Set3)
    fig_gender.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_gender, use_container_width=True)
    
    st.subheader("Gender by Purchase Segment")
    gender_segment = df_filtered.groupby(['Gender','Purchase_Segment']).size().reset_index(name='Count')
    fig_segment = px.bar(gender_segment, x='Gender', y='Count', color='Purchase_Segment', barmode='group', color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_segment, use_container_width=True)

with tab3:
    st.subheader("Purchase Segment Funnel")
    segment_counts = df_filtered['Purchase_Segment'].value_counts()
    fig_funnel = px.funnel(y=segment_counts.index, x=segment_counts.values)
    st.plotly_chart(fig_funnel, use_container_width=True)
    
    st.subheader("Purchase Count Distribution")
    fig_hist = px.histogram(df_filtered, x='Purchase_Count', nbins=30, color_discrete_sequence=['#636EFA'])
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Purchase Trends Over Time")
    monthly_purchases = df_filtered.groupby('Purchase_Month').agg({'Purchase_Count':['sum','count']}).reset_index()
    monthly_purchases.columns = ['Month', 'Total_Purchases', 'Customer_Count']
    fig_trends = make_subplots(specs=[[{"secondary_y": True}]])
    fig_trends.add_trace(go.Scatter(x=monthly_purchases['Month'], y=monthly_purchases['Total_Purchases'], name='Total Purchases', line=dict(color='#636EFA', width=3)), secondary_y=False)
    fig_trends.add_trace(go.Scatter(x=monthly_purchases['Month'], y=monthly_purchases['Customer_Count'], name='Customer Count', line=dict(color='#EF553B', width=3, dash='dash')), secondary_y=True)
    fig_trends.update_xaxes(title_text='Month')
    fig_trends.update_yaxes(title_text='Total Purchases', secondary_y=False)
    fig_trends.update_yaxes(title_text='Customer Count', secondary_y=True)
    st.plotly_chart(fig_trends, use_container_width=True)

with tab4:
    st.subheader("Feedback Score Distribution")
    fig_feedback_dist = px.histogram(df_filtered, x='Feedback_Score', nbins=20, color_discrete_sequence=['#00CC96'])
    st.plotly_chart(fig_feedback_dist, use_container_width=True)

    st.subheader("Feedback Score by Gender")
    fig_feedback_gender = px.violin(df_filtered, y='Feedback_Score', x='Gender', color='Gender', box=True)
    st.plotly_chart(fig_feedback_gender, use_container_width=True)

    if 'Feedback_Text' in df_filtered.columns:
        st.subheader("Sentiment Distribution")
        sentiment_counts = df_filtered['Sentiment_Category'].value_counts()
        fig_sentiment = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig_sentiment, use_container_width=True)

with tab5:
    st.subheader("Top 10 States by Customer Count")
    top_states = df_filtered['State'].value_counts().head(10)
    fig_states = px.bar(x=top_states.index, y=top_states.values, title="Top States", labels={'x':'State','y':'Customers'}, color=top_states.values, color_continuous_scale='Blues')
    st.plotly_chart(fig_states, use_container_width=True)

    st.subheader("Top 10 States by Avg Feedback")
    state_feedback = df_filtered.groupby('State')['Feedback_Score'].mean().sort_values(ascending=False).head(10)
    fig_state_feedback = px.bar(x=state_feedback.index, y=state_feedback.values, title="Top States by Feedback", color=state_feedback.values, color_continuous_scale='Oranges')
    st.plotly_chart(fig_state_feedback,use_container_width=True)

    st.subheader("City-wise Customer Distribution")
    top_cities = df_filtered['City'].value_counts().head(15)
    fig_cities = px.treemap(names=top_cities.index, parents=['']*len(top_cities), values=top_cities.values, title="Top Cities - Customer Distribution")
    st.plotly_chart(fig_cities, use_container_width=True)

with tab6:
    st.subheader("Churn Prediction (Sample Model)")
    churn_df = df_filtered.dropna(subset=['Feedback_Score','Purchase_Count'])
    if len(churn_df) > 0:
        X = churn_df[['Feedback_Score','Purchase_Count','Monetary']]
        y = np.where(churn_df['Purchase_Segment']=='Low',1,0)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X,y)
        churn_df['Churn_Probability'] = model.predict_proba(X)[:,1]
        st.dataframe(churn_df[['Name','Churn_Probability']].head(15))
        fig_churn = px.histogram(churn_df, x='Churn_Probability', nbins=20, title='Churn Probability Distribution')
        st.plotly_chart(fig_churn, use_container_width=True)
    else:
        st.info("Not enough data for churn prediction.")

st.divider()
st.markdown("<center>📊 Enhanced Customer Insights Analyzer - 2025 Edition</center>", unsafe_allow_html=True)
