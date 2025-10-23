# ==========================================
# Enhanced Customer Insights Analyzer
# Modern Tech Stack - Streamlit Cloud Ready
# ==========================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gender_guesser.detector as gender
import os
import warnings
from datetime import datetime
import numpy as np

warnings.filterwarnings("ignore")

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Customer Insights Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CACHING FOR PERFORMANCE
# ==========================================
@st.cache_data
def load_data(file_path=None, uploaded_file=None):
    """Load and cache data for better performance"""
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, parse_dates=['Last_Purchase_Date'], dayfirst=True)
        elif file_path and os.path.exists(file_path):
            df = pd.read_csv(file_path, parse_dates=['Last_Purchase_Date'], dayfirst=True)
        else:
            return None
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def process_data(df):
    """Process and transform data with caching"""
    # Name processing
    df['Name'] = df['Name'].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
    df['Name'] = df['Name'].str.replace(r'\xa0', ' ', regex=True)
    df[['First_Name', 'Last_Name']] = df['Name'].str.split(' ', n=1, expand=True)
    df['Last_Name'] = df['Last_Name'].fillna('')
    
    # Gender detection
    d = gender.Detector()
    df['Gender'] = df['First_Name'].apply(lambda x: d.get_gender(x))
    df['Gender'] = df['Gender'].replace({
        'male': 'Male',
        'mostly_male': 'Male',
        'female': 'Female',
        'mostly_female': 'Female',
        'andy': 'Unknown'
    })
    
    # Purchase segments
    df['Purchase_Segment'] = pd.cut(
        df['Purchase_Count'],
        bins=[0, 5, 12, float('inf')],
        labels=['Low', 'Medium', 'High']
    )
    
    # Date processing
    df['Last_Purchase_Date'] = pd.to_datetime(df['Last_Purchase_Date'], errors='coerce')
    df['Purchase_Month'] = df['Last_Purchase_Date'].dt.to_period('M').astype(str)
    df['Purchase_Year'] = df['Last_Purchase_Date'].dt.year
    
    return df

# ==========================================
# MAIN APP
# ==========================================
st.markdown('<h1 class="main-header">📊 Customer Insights Analyzer</h1>', unsafe_allow_html=True)
st.markdown("### Analyze customer demographics, gender distribution, and feedback trends with interactive visualizations!")

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.title("⚙️ Configuration")
    
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"], help="Upload your customer data CSV file")
    default_file_path = os.path.join(os.getcwd(), "2000_dataset.csv")
    
    st.divider()
    
    # Filter options
    st.subheader("🔍 Filters")

# ==========================================
# LOAD AND PROCESS DATA
# ==========================================
df = load_data(default_file_path, uploaded_file)

if df is None:
    st.error("❌ No dataset available! Please upload a CSV file.")
    st.stop()

if uploaded_file:
    st.success("✅ File uploaded successfully!")
else:
    st.info(f"⚡ Using default dataset")

# Process data
df = process_data(df)

# ==========================================
# SIDEBAR FILTERS
# ==========================================
with st.sidebar:
    gender_filter = st.multiselect(
        "Select Gender",
        options=df['Gender'].unique(),
        default=df['Gender'].unique()
    )
    
    state_filter = st.multiselect(
        "Select States",
        options=sorted(df['State'].unique()),
        default=df['State'].unique()
    )
    
    purchase_segment_filter = st.multiselect(
        "Purchase Segment",
        options=['Low', 'Medium', 'High'],
        default=['Low', 'Medium', 'High']
    )

# Apply filters
df_filtered = df[
    (df['Gender'].isin(gender_filter)) &
    (df['State'].isin(state_filter)) &
    (df['Purchase_Segment'].isin(purchase_segment_filter))
]

# ==========================================
# KEY METRICS DASHBOARD
# ==========================================
st.header("📈 Key Metrics")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="Total Customers",
        value=f"{len(df_filtered):,}",
        delta=f"{len(df_filtered) - len(df)}" if len(df_filtered) != len(df) else None
    )

with col2:
    st.metric(
        label="Avg Feedback Score",
        value=f"{df_filtered['Feedback_Score'].mean():.2f}",
        delta=f"{(df_filtered['Feedback_Score'].mean() - df['Feedback_Score'].mean()):.2f}"
    )

with col3:
    st.metric(
        label="Total Purchases",
        value=f"{df_filtered['Purchase_Count'].sum():,}"
    )

with col4:
    st.metric(
        label="Avg Purchase Count",
        value=f"{df_filtered['Purchase_Count'].mean():.1f}"
    )

with col5:
    st.metric(
        label="States Covered",
        value=f"{df_filtered['State'].nunique()}"
    )

st.divider()

# ==========================================
# TABBED INTERFACE
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Data Overview",
    "👥 Demographics",
    "📊 Purchase Analysis",
    "⭐ Feedback Analysis",
    "📍 Geographic Insights"
])

# ==========================================
# TAB 1: DATA OVERVIEW
# ==========================================
with tab1:
    st.subheader("📋 Dataset Preview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(
            df_filtered.head(100),
            use_container_width=True,
            height=400
        )
    
    with col2:
        st.subheader("📊 Data Statistics")
        st.write(df_filtered.describe())
    
    # Download button
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data",
        data=csv,
        file_name=f"customer_insights_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# ==========================================
# TAB 2: DEMOGRAPHICS
# ==========================================
with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👥 Gender Distribution")
        gender_counts = df_filtered['Gender'].value_counts()
        fig_gender = px.pie(
            values=gender_counts.values,
            names=gender_counts.index,
            title="Gender Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.4
        )
        fig_gender.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_gender, use_container_width=True)
    
    with col2:
        st.subheader("📊 Gender by Purchase Segment")
        gender_segment = df_filtered.groupby(['Gender', 'Purchase_Segment']).size().reset_index(name='Count')
        fig_segment = px.bar(
            gender_segment,
            x='Gender',
            y='Count',
            color='Purchase_Segment',
            title="Customer Distribution by Gender and Purchase Segment",
            barmode='group',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_segment, use_container_width=True)
    
    # Age distribution (if available)
    st.subheader("🎂 Purchase Behavior by Gender")
    fig_box = px.box(
        df_filtered,
        x='Gender',
        y='Purchase_Count',
        color='Gender',
        title="Purchase Count Distribution by Gender",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    st.plotly_chart(fig_box, use_container_width=True)

# ==========================================
# TAB 3: PURCHASE ANALYSIS
# ==========================================
with tab3:
    st.subheader("🛒 Purchase Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Purchase segments distribution
        segment_counts = df_filtered['Purchase_Segment'].value_counts()
        fig_segments = px.funnel(
            y=segment_counts.index,
            x=segment_counts.values,
            title="Customer Funnel by Purchase Segment"
        )
        st.plotly_chart(fig_segments, use_container_width=True)
    
    with col2:
        # Purchase count histogram
        fig_hist = px.histogram(
            df_filtered,
            x='Purchase_Count',
            nbins=30,
            title="Purchase Count Distribution",
            color_discrete_sequence=['#636EFA']
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Purchase trends over time
    st.subheader("📈 Purchase Trends Over Time")
    monthly_purchases = df_filtered.groupby('Purchase_Month').agg({
        'Purchase_Count': ['sum', 'count']
    }).reset_index()
    monthly_purchases.columns = ['Month', 'Total_Purchases', 'Customer_Count']
    
    fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
    fig_trend.add_trace(
        go.Scatter(x=monthly_purchases['Month'], y=monthly_purchases['Total_Purchases'],
                   name="Total Purchases", line=dict(color='#636EFA', width=3)),
        secondary_y=False
    )
    fig_trend.add_trace(
        go.Scatter(x=monthly_purchases['Month'], y=monthly_purchases['Customer_Count'],
                   name="Customer Count", line=dict(color='#EF553B', width=3, dash='dash')),
        secondary_y=True
    )
    fig_trend.update_xaxes(title_text="Month")
    fig_trend.update_yaxes(title_text="Total Purchases", secondary_y=False)
    fig_trend.update_yaxes(title_text="Customer Count", secondary_y=True)
    st.plotly_chart(fig_trend, use_container_width=True)

# ==========================================
# TAB 4: FEEDBACK ANALYSIS
# ==========================================
with tab4:
    st.subheader("⭐ Feedback Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Feedback score distribution
        fig_feedback_dist = px.histogram(
            df_filtered,
            x='Feedback_Score',
            nbins=20,
            title="Feedback Score Distribution",
            color_discrete_sequence=['#00CC96']
        )
        st.plotly_chart(fig_feedback_dist, use_container_width=True)
    
    with col2:
        # Feedback by gender
        fig_feedback_gender = px.violin(
            df_filtered,
            y='Feedback_Score',
            x='Gender',
            color='Gender',
            box=True,
            title="Feedback Score Distribution by Gender"
        )
        st.plotly_chart(fig_feedback_gender, use_container_width=True)
    
    # Top cities by feedback
    st.subheader("🏙️ Top 10 Cities by Average Feedback")
    top_cities_feedback = df_filtered.groupby('City')['Feedback_Score'].mean().sort_values(ascending=False).head(10)
    fig_cities = px.bar(
        x=top_cities_feedback.values,
        y=top_cities_feedback.index,
        orientation='h',
        title="Top 10 Cities by Average Feedback Score",
        color=top_cities_feedback.values,
        color_continuous_scale='Viridis',
        labels={'x': 'Average Feedback Score', 'y': 'City'}
    )
    st.plotly_chart(fig_cities, use_container_width=True)
    
    # Feedback over time
    st.subheader("📅 Feedback Trends Over Time")
    feedback_time = df_filtered.groupby('Purchase_Month')['Feedback_Score'].mean().reset_index()
    fig_feedback_time = px.line(
        feedback_time,
        x='Purchase_Month',
        y='Feedback_Score',
        title="Average Feedback Score Over Time",
        markers=True,
        line_shape='spline'
    )
    fig_feedback_time.update_traces(line_color='#FF6692', line_width=3)
    st.plotly_chart(fig_feedback_time, use_container_width=True)
    
    # Correlation: Purchase Count vs Feedback
    st.subheader("🔗 Purchase Count vs Feedback Score")
    fig_scatter = px.scatter(
        df_filtered,
        x='Purchase_Count',
        y='Feedback_Score',
        color='Gender',
        size='Purchase_Count',
        title="Purchase Count vs Feedback Score by Gender",
        trendline="ols",
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ==========================================
# TAB 5: GEOGRAPHIC INSIGHTS
# ==========================================
with tab5:
    st.subheader("🗺️ Geographic Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top states
        st.subheader("🏆 Top 10 States by Customer Count")
        top_states = df_filtered['State'].value_counts().head(10)
        fig_states = px.bar(
            x=top_states.index,
            y=top_states.values,
            title="Top 10 States",
            labels={'x': 'State', 'y': 'Customer Count'},
            color=top_states.values,
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_states, use_container_width=True)
    
    with col2:
        # State-wise average feedback
        st.subheader("⭐ Top 10 States by Feedback")
        state_feedback = df_filtered.groupby('State')['Feedback_Score'].mean().sort_values(ascending=False).head(10)
        fig_state_feedback = px.bar(
            x=state_feedback.index,
            y=state_feedback.values,
            title="Top 10 States by Average Feedback",
            labels={'x': 'State', 'y': 'Average Feedback'},
            color=state_feedback.values,
            color_continuous_scale='Oranges'
        )
        st.plotly_chart(fig_state_feedback, use_container_width=True)
    
    # City analysis
    st.subheader("🏙️ City-wise Customer Distribution")
    top_cities = df_filtered['City'].value_counts().head(15)
    fig_cities_count = px.treemap(
        names=top_cities.index,
        parents=[''] * len(top_cities),
        values=top_cities.values,
        title="Top 15 Cities - Customer Distribution"
    )
    st.plotly_chart(fig_cities_count, use_container_width=True)

# ==========================================
# FOOTER
# ==========================================
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>📊 Customer Insights Analyzer v2.0 | Built with Streamlit & Plotly</p>
    <p>💡 Enhanced with modern visualizations and performance optimization</p>
</div>
""", unsafe_allow_html=True)

st.success("✅ Analysis Complete! Use the tabs above to explore different insights.")
