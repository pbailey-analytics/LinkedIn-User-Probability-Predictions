#import packages here....
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import altair as alt

st.set_page_config(page_title="LinkedIn Use Probability",layout= "wide")

#1) Read in the data, call the dataframe "s"  and check the dimensions of the dataframe
s = pd.read_csv("social_media_usage.csv")

#2) create function
def clean_sm(x):
    x = np.where(x ==1,1,0)
    return x

# Create ss dataframe
ss = pd.DataFrame()
ss["sm_li"] = clean_sm(s["web1h"]) # linkedIn column

# Apply missing rules for other predictors
ss["income"] = s["income"].where(s["income"] <= 9)
ss["educ2"] = s["educ2"].where(s["educ2"] <= 8)
ss["par"] = s["par"]
ss["marital"] = s["marital"]
ss["gender"] = s["gender"]
ss["age"] = s["age"].where(s["age"] <= 98)

# Drop rows with missing values
ss = ss.dropna().reset_index(drop=True)

#Q4 create a target vector (y) and feature set (X)

# Target
y = ss["sm_li"]

#feature set
X = ss[["income","educ2","par","marital","gender","age"]]

# Split into training and test sets (80/20), stratify on sm_li
X_train, X_test,y_train, y_test, = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=385,
    stratify=y)

# Instantiate logistic regression with class_weight balanced
log_reg = LogisticRegression(
    class_weight="balanced",
    random_state=385,   
    max_iter=10000)    

log_reg.fit(X_train, y_train)

feature_cols = X_train.columns

# Q9 – Scenario predictions using fitted model updated for Streamlit.

st.title("LinkedIn User Probability Predictions")

col1, col2 = st.columns(2)


# 1. Inputs
with col1:
    with st.container():
        st.markdown("### Profile Characteristics")
        income = st.slider("Income (1–9)", 1, 9, 8)
        education = st.slider("Education (1–8)", 1, 8, 7)
        parent = st.checkbox("Parent?", value=False)
        married = st.checkbox("Married?", value=True)
        female = st.checkbox("Female?", value=True)
 
        
with col2:
    with st.container():
        st.markdown("### Prediction Settings")
        age1 = st.slider("Age of person 1", 18, 98, 42)
        age2 = st.slider("Age of person 2", 18, 98, 82)
        user_decision = st.slider(
        "LinkedIn User Classification",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="If the predicted probability is above this value, the person is classified as a LinkedIn user.")
        
 # person 1
person1 = pd.DataFrame([{
    "income": income,
    "educ2": education,
    "par": parent,      
    "marital": married,
    "gender": female,
    "age": age1
}])[feature_cols]

# Person 2
person2 = pd.DataFrame([{
    "income": income,
    "educ2": education,
    "par": parent,
    "marital": married,
    "gender": female,
    "age": age2
}])[feature_cols]

# Button: Calculates probability and runs the chart.
if st.button("Run LinkedIn prediction for both ages"):

# Person 1 prob. and user decision calc.
    prob1 = log_reg.predict_proba(person1)[:, 1][0]
    label1 = "LinkedIn user" if prob1 >= user_decision else "Non-user"

    col3, col4 = st.columns(2)

    with col3:
        st.subheader(f"Person 1 (Age {age1})")
        st.write(f"Probability LinkedIn user: **{prob1 * 100:.1f}%**")
        st.write(f"Classification: **{label1}**")

# Person 2 prob. and user decision calc.
    prob2 = log_reg.predict_proba(person2)[:, 1][0]
    label2 = "LinkedIn user" if prob2 >= user_decision else "Non-user"

    with col4:
        st.subheader(f"Person 2 (Age {age2})")
        st.write(f"Probability LinkedIn user: **{prob2 * 100:.1f}%**")
        st.write(f"Classification: **{label2}**")

    st.divider()

    st.subheader("Probability of LinkedIn Use by Age")

    chart_df = pd.DataFrame({
            "Age": [age1, age2],
            "Probability (%)": [prob1 * 100, prob2 * 100],})

    chart = (
        alt.Chart(chart_df)
        .mark_bar(color="#003B5C")  
        .encode(
            x=alt.X("Age:O", axis=alt.Axis(labelAngle=0, title="Age")),
            y=alt.Y(
                "Probability (%):Q",
                title="Predicted Probability (%)",
                scale=alt.Scale(domain=[0, 100]),
            ),
            tooltip=[
                alt.Tooltip("Age:O", title="Age"),
                alt.Tooltip("Probability (%):Q", format=".1f", title="Probability (%)"),],))

    st.altair_chart(chart, use_container_width=True)