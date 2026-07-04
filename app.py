import streamlit as st
import numpy as np
import plotly.express as px
import pandas as pd
import pickle


model = pickle.load(open("model.pkl","rb"))



df = pd.read_csv("Cleaned_data.csv")
st.markdown("""
            <style>
            .stApp{
            background: linear-gradient(to top, #D3B6E0, #ffffff);
            }
       </style>""",unsafe_allow_html=True)



st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #542966;
    color:white;
            
}
</style>
""", unsafe_allow_html=True)



st.sidebar.markdown("""
<style>
.circle-img {
    width: 120px;
    height: 120px;
    border-radius: 50%;
    object-fit: cover;
    display: block;
    margin-top: 0px;
    margin-bottom: 10px;
    margin-left : 40px
   
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* text input box */
div[data-testid="stTextInput"] input {
    background-color: #f3e5f5;
    color: #D3B6E0;
    border: 2px solid #542966;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* label color for all inputs */
label {
    color: #D3B6E0 !important;
    font-weight: bold;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown(
    '<img src="https://thumbs.dreamstime.com/b/diabetes-care-vector-icon-design-blue-circle-emblem-eps-available-90119177.jpg" class="circle-img">',
    unsafe_allow_html=True
)


st.sidebar.markdown("""
        <h2 style="color:#D3B6E0; "><u> User Info</u> </h2>
  """,unsafe_allow_html = True)


name = st.sidebar.text_input("Enter username")
st.sidebar.write("Username : " ,name)

st.sidebar.markdown("""
        <h2 style="color:#D3B6E0; "><u> Model</u> </h2>
  """,unsafe_allow_html = True)
st.sidebar.write("Algorithm: XGBoost")
st.sidebar.write("Accuracy: ~97%")

st.sidebar.markdown("""
        <h2 style="color:#D3B6E0; "><u> About</u> </h2>
  """,unsafe_allow_html = True)
st.sidebar.write("Predicts diabetes risk using ML")

st.sidebar.markdown("""
        <h2 style="color:#D3B6E0; "><u> Developer</u> </h2>
  """,unsafe_allow_html = True)
st.sidebar.write("Created by: Vishakha Nikam")


st.markdown("""
<h1 style='text-align: center; font-size: 42px; color: #542966;'>
🏥 Diabetes Patient Prediction App
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Selectbox label only */
div[data-testid="stSelectbox"] label {
    color: #542966 !important;
    font-weight: bold;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)


gender = st.selectbox("Gender", ["Male", "Female"])

hypertension = st.selectbox("Hypertension", ["No", "Yes"])

heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])

smoking_history = st.selectbox(
    "Smoking History",
    ["never", "former", "current", "not current", "ever"]
)


st.markdown("""
<style>
/* Number input label only */
div[data-testid="stNumberInput"] label {
    color: #542966 !important;
    font-weight: bold;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

age = st.number_input("Age", min_value=0, max_value=120, value=25)

bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, value=22.5)

hba1c = st.number_input("HbA1c Level", min_value=0.0, max_value=20.0, value=5.5)

glucose = st.number_input(
    "Blood Glucose Level",
    min_value=0,
    max_value=300,
    value=100
)


st.markdown("""  
   <style>
            div.stButton > button {
    background-color: #542966;
    color: white;
    font-size: 30px;
    font-weight: bold;
    width: 200px;
    height: 55px;
    border-radius: 12px;
    border: none;
}

div.stButton > button:active {
    background-color: #542966;
}
            </style>
 """,unsafe_allow_html=True)



if st.button("Predict"):

    gender_map = {"Male": 1, "Female": 0}
    hypertension_map = {"No": 0, "Yes": 1}
    heart_map = {"No": 0, "Yes": 1}
    smoking_map = {"never":0, "former":1, "current":2, "not current":3, "ever":4}

    gender_val = gender_map[gender]
    hypertension_val = hypertension_map[hypertension]
    heart_val = heart_map[heart_disease]
    smoking_val = smoking_map[smoking_history]

    input_data = np.array([[
        gender_val,
        age,
        hypertension_val,
        heart_val,
        smoking_val,
        bmi,
        hba1c,
        glucose
    ]])
     
   

    prediction = model.predict(input_data)
    
    st.markdown("""
    <hr>
    """, unsafe_allow_html=True)

    st.markdown("""
    <h4 style="
        background-color:#753F8C;
        padding:20px;
        border-radius:10px;
        color:#ffffff;
        font-size:20px;
        font-weight:bold;
        text-align:center;
        margin-top : 20px;
        margin-bottom: 20px;
    ">
    Prediction Result
    </h4>
    """, unsafe_allow_html=True)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")

    


    st.subheader("📊 Patient Input Summary")

    data = pd.DataFrame({
        "Feature": [
            'gender', 'age', 'hypertension', 'heart_disease', 'smoking_history',
       'bmi', 'HbA1c_level', 'blood_glucose_level'
        ],
        "Value": [
           gender,hypertension ,heart_disease,smoking_history,age,bmi,hba1c,glucose
        ]
    })

    fig = px.bar(
        data,
        x="Feature",
        y="Value",
        color="Value",
        color_continuous_scale="Blues",
        text="Value"
    )

    fig.update_layout(
        title="Patient Health Parameters", 
        xaxis_title="Features",
        yaxis_title="Values",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)
