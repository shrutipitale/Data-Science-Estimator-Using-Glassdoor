import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(page_title="Salary Estimator", page_icon=":moneybag:", layout="wide")

with st.container():
        st.title("Data Science Estimator Using Glassdoor")
        st.subheader("Project Overview")
                # Section: Learn About Data
        st.write(""" The data in the dataset is extracted from the Glassdoor website, which is a job posting website. The dataset has data related to data science jobs and salaries and a lot more, offering a clear view of job opportunities. 
        It is packed with essential details like job titles, estimated salaries, job descriptions, company ratings,  and key company info such as location, size, and industry.Whether you're job hunting or researching, this dataset helps you understand the job market easily. 
        Start exploring now to make smart career choices! """)
        
        st.write("""Perfect for adding to your Kaggle notebooks, our dataset is a treasure trove for analyzing all kinds of job-related info.Whether you're curious about salary trends or want to find the best-rated companies, this dataset has you covered. 
        It's great for beginners and experts alike, offering lots of chances to learn and discover.You can use it to predict things or find hidden patterns—there's so much you can do! 
        So, get ready to explore the world of jobs with our easy-to-use dataset on Kaggle.""")
        st.write(" Optimized Lasso Regession and RandomForestRegressors using RandomsearchCV to reach the best model.")

        st.markdown(":link:[**Github Repository**](https://github.com/shrutipitale/Data-Science-Estimator-Using-Glassdoor.git)")
        

 # Create two equal-width columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 TABLE OF CONTENTS 📋")
    st.markdown("""
    1. IMPORTING LIBRARIES  
    2. LOADING DATA  
    3. DATA CLEANING  
    4. Exploratory Data Analysis  
    5. Model Building  
    6. TUNING BY GRIDSEARCHCV  
    7. TEST ENSEMBLING  
    8. PUTTING MODEL INTO PRODUCTION  
    """)

with col2:
    st.subheader("🔄  Data Science Estimator Using Glassdoor Project 🤖")
    st.markdown("""
    1. Understanding the Problem Statement  
    2. Data Checks to Perform  
    3. Exploratory Data Analysis  
    4. Data Pre-Processing  
    5. Model Training  
    6. Choose Best Model  
    7. Model Tuning  
    8. Test Ensembling  
    9. Putting Model into Production  
    """)

st.write('----')
st.subheader("Enter values to estimate the salary")

with st.container():
    left_col, mid_col, right_col = st.columns(3)
    with left_col:
        st.subheader("Choose the size of company")
        num_of_employes = st.selectbox("Number of employees", [
            'Unknown', '10000+ employees', '5001 to 10000 employees',
            '1001 to 5000 employees', '501 to 1000 employees',
            '201 to 500 employees', '51 to 200 employees',
            '1 to 50 employees'
        ], index=0)

    with mid_col:
        st.subheader("Choose the Type of ownership")
        company_type = st.selectbox("Type of company", [
            'Unknown', 'Private', 'Other Organization', 'Government',
            'Public', 'Hospital', 'Subsidiary or Business Segment',
            'Nonprofit Organization', 'College/University',
            'School/School District'
        ], index=0)

    with right_col:
        st.subheader("Choose job role")
        job_role = st.selectbox("Job role", [
             'Unknown','data scientist', 'data analyst', 'data engineer',
            'data science director', 'data science manager',
            'machine learning engineer'
        ])

with st.container():
    left_column, mid_column, right_column = st.columns(3)
    with left_column:
        st.subheader("Choose the level you are offered")
        seniority = st.selectbox("Level", ['na', 'senior', 'jr'])

    with mid_column:
        st.subheader("How old is the company")
        age = st.slider("Years", min_value=1, max_value=100, value=10, step=2)

    with right_column:
        st.empty()

# ------------------- Backend --------------------------

# Initialize all features with zeros
features = [0] * 30  # Total features model expects

# Set age (feature 0)
features[0] = age

# Company size (features 1-8)
company_size_map = {
    '10000+ employees': 1,
    '5001 to 10000 employees': 2,
    '1001 to 5000 employees': 3,
    '501 to 1000 employees': 4,
    '201 to 500 employees': 5,
    '51 to 200 employees': 6,
    '1 to 50 employees': 7
}
if num_of_employes in company_size_map:
    features[company_size_map[num_of_employes]] = 1

# Ownership type (features 9-17)
ownership_map = {
    'Private': 9,
    'Other Organization': 10,
    'Government': 11,
    'Public': 12,
    'Hospital': 13,
    'Subsidiary or Business Segment': 14,
    'Nonprofit Organization': 15,
    'College/University': 16,
    'School/School District': 17
}
if company_type in ownership_map:
    features[ownership_map[company_type]] = 1

# Job role (features 18-23)
job_role_map = {
    'data scientist': 18,
    'data analyst': 19,
    'data engineer': 20,
    'data science director': 21,
    'data science manager': 22,
    'machine learning engineer': 23
}
if job_role in job_role_map:
    features[job_role_map[job_role]] = 1

# Seniority (features 24-25)
seniority_map = {
    'senior': 24,
    'jr': 25
}
if seniority in seniority_map:
    features[seniority_map[seniority]] = 1

# Convert to numpy array and reshape
data = np.array(features).reshape(1, -1)

# Prediction button
try:
    with open('Salary_predictions.pkl', 'rb') as file:
        model = pickle.load(file)

    if st.button("Predict"):
        try:
            pred = model.predict(data)[0]
            st.success(f"The estimated salary is ${round(pred, 2)}k/year")
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.write("Input data shape:", data.shape)
            st.write("Input data:", data)
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
