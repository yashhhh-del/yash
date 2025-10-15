import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gender_guesser.detector as gender
import os

sns.set(style="whitegrid")

st.title("📊 Customer Name Insights Analyzer")
st.write("Analyze customer demographics, gender distribution, and feedback trends!")

# ------------------------------
# File Upload / Default
# ------------------------------

uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])

# Default file location (same folder as app)
default_file_path = os.path.join(os.getcwd(), "2000_dataset.csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=['Last_Purchase_Date'], dayfirst=True)
    st.success("✅ File uploaded successfully!")
elif os.path.exists(default_file_path):
    df = pd.read_csv(default_file_path, parse_dates=['Last_Purchase_Date'], dayfirst=True)
    st.info(f"⚡ No file uploaded. Using default dataset: `{default_file_path}`")
else:
    st.error("❌ No dataset available! Please upload a CSV file.")
    st.stop()

# ------------------------------
# Preview Data
# ------------------------------
st.subheader("Preview of Data")
st.write(df.head())

# ------------------------------
# Name Split
# ------------------------------
df[['First_Name', 'Last_Name']] = df['Name'].str.split(' ', n=1, expand=True)

# ------------------------------
# Gender Detection
# ------------------------------
d = gender.Detector()
df['Gender'] = df['First_Name'].apply(lambda x: d.get_gender(x))
df['Gender'] = df['Gender'].replace({
    'male': 'Male',
    'mostly_male': 'Male',
    'female': 'Female',
    'mostly_female': 'Female',
    'andy': 'Unknown'
})

# ------------------------------
# Gender Distribution Pie Chart
# ------------------------------
st.subheader("Gender Distribution")
fig1, ax1 = plt.subplots(figsize=(6, 6))
df['Gender'].value_counts().plot.pie(
    autopct='%1.1f%%',
    colors=['#66b3ff', '#ff9999', '#99ff99'],
    startangle=90,
    ax=ax1
)
st.pyplot(fig1)

# ------------------------------
# Top States by Customer Count
# ------------------------------
st.subheader("Top 5 States by Customer Count")
top_states = df['State'].value_counts().head(5)
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.barplot(x=top_states.index, y=top_states.values, palette='viridis', ax=ax2, hue=top_states.index, legend=False)
st.pyplot(fig2)

# ------------------------------
# Purchase Segment
# ------------------------------
def purchase_segment(x):
    if x <= 5:
        return 'Low'
    elif x <= 12:
        return 'Medium'
    else:
        return 'High'

df['Purchase_Segment'] = df['Purchase_Count'].apply(purchase_segment)

# ------------------------------
# Feedback per City
# ------------------------------
st.subheader("Average Feedback Score per City")
feedback_city = df.groupby('City')['Feedback_Score'].mean().sort_values(ascending=False)
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.barplot(x=feedback_city.index, y=feedback_city.values, palette='magma', ax=ax3, hue=feedback_city.index, legend=False)
plt.xticks(rotation=45)
st.pyplot(fig3)

# ------------------------------
# Feedback Over Time
# ------------------------------
st.subheader("Average Feedback Over Time")
df['Last_Purchase_Date'] = pd.to_datetime(df['Last_Purchase_Date'], errors='coerce')
feedback_time = df.groupby(df['Last_Purchase_Date'].dt.to_period('M'))['Feedback_Score'].mean()
fig4, ax4 = plt.subplots(figsize=(12, 6))
feedback_time.plot(kind='line', marker='o', ax=ax4)
st.pyplot(fig4)

# ------------------------------
# Scatter Plot Purchase vs Feedback
# ------------------------------
st.subheader("Purchase Count vs Feedback Score (by Gender)")
fig5, ax5 = plt.subplots(figsize=(8, 6))
sns.scatterplot(x='Purchase_Count', y='Feedback_Score', hue='Gender', data=df, palette='Set1', ax=ax5)
st.pyplot(fig5)

st.success("✅ Analysis Complete!")
