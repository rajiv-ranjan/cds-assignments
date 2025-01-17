import pickle
import Orange
from Orange.data import Domain, DiscreteVariable
import streamlit as st
import numpy as np
import pandas as pd
import shap
from streamlit_shap import st_shap

#LOAD TRAINED MODEL
with open("titanic_rforest.pkcls", "rb") as model:
    loaded_model = pickle.load(model)

#ADD COMPONENTS AND SIDEBAR
side_bar = st.sidebar
header = st.container()
body = st.container()
footer = st.container()

#HANDLE DATA AND GET INPUT
@st.cache_data
def get_data():
    data = pd.read_csv("titanic.tab",sep="\t")
    data = data.drop([0, 1])
    return data

def get_user_input():
    with side_bar:
        user_status = st.radio("status",("crew", "first", "second", "third"))
        user_age = st.radio("age", ("adult", "child"))
        user_sex = st.radio("sex", ("male", "female"))
    
    domain = Domain([DiscreteVariable("status", values=["crew", "first", "second", "third"]), DiscreteVariable("age", values=["adult", "child"]),\
                      DiscreteVariable("sex", values=["male", "female"])])
    status = ["crew", "first", "second", "third"]
    status_inp = status.index(user_status)
    age = ["adult", "child"]
    age_inp = age.index(user_age)
    sex = ["male", "female"]
    sex_inp = sex.index(user_sex)
    X = np.column_stack((status_inp, age_inp, sex_inp))
    data_table = Orange.data.Table(domain, X)
    
    return [user_status, user_age, user_sex], data_table

def input_to_df(input_values):
    status_crew = 1.0 if "crew" in input_values else 0.0
    status_first = 1.0 if "first" in input_values else 0.0
    status_second = 1.0 if "second" in input_values else 0.0
    status_third = 1.0 if "third" in input_values else 0.0
    age_adult = 1.0 if "adult" in input_values else 0.0
    age_child = 1.0 if "child" in input_values else 0.0
    sex_female = 1.0 if "female" in input_values else 0.0
    sex_male = 1.0 if "male" in input_values else 0.0
    df = pd.DataFrame({
        "status_crew": status_crew,
        "status_first": status_first,
        "status_second": status_second,
        "status_third": status_third,
        "age_adult": age_adult,
        "age_child": age_child,
        "sex_female": sex_female,
        "sex_male": sex_male
    }, index=[0])
    return df

def make_force_plot(input_values, data_table):
    skl_model = loaded_model.skl_model
    data = get_data()
    features = ['status', 'age', 'sex']
    X = pd.get_dummies(data[features])
    input_df = input_to_df(input_values)
    explainer = shap.TreeExplainer(skl_model)
    shap_values = explainer.shap_values(input_df)
    plt = shap.force_plot(explainer.expected_value[1], shap_values[1], input_df)
    return plt


#INPUT
with side_bar:
    st.markdown("## Choose who is boarding the Titanic")
#DATASET INFO
with header:
    st.title("Titanic Survival Prediction")
    st.markdown("This is an app which predicts whether a passenger on the Titanic survived the famous shipwreck based on different features.")
    st.markdown("Features defined in our dataset are:- ")
    st.markdown("* Status")
    st.markdown("* Age")
    st.markdown("* Sex")

    with st.expander("Show Dataset"):
        st.dataframe(get_data())

    st.divider()
#USER INPUT 
with body:
    st.markdown("### Will your passenger survive the Titanic shipwreck?")
    user_input, data_table = get_user_input()
    input_col, result_col = st.columns(2)
    with input_col:
        st.markdown("##### Your Input: ")
        st.markdown(f"Status: `{user_input[0]}`")
        st.markdown(f"Age: `{user_input[1]}`")
        st.markdown(f"Sex: `{user_input[2]}`")
    with result_col:
        st.markdown("##### Prediction: ")
        prediction = loaded_model(data_table)
        if prediction:
            st.markdown("Your passenger **_will_** survive")
        else:
            st.markdown("Your passenger **_will not_** survive")
    st.divider()
#MODEL ACCURACY AND OTHER MEASURES 
with footer:
    st.markdown("### SHAP analysis for prediction: ")
    fig = make_force_plot(user_input, data_table)
    st_shap(fig)
    with st.expander("Explanation of plot"):
        st.markdown("The plot shows each feature's contribution towards the prediction with values closer to 1 ascertaining survival \
                    while those closer to 0 showing slimmer chances of making it out of the shipwreck.")
        st.markdown("* Features in red increase the value of prediction and the greater the length of the bar,\
                     the greater is the feature's contribution in prediction.")
        st.markdown("* Features in blue do the reverse. They decrease the chances of survival and their weight in prediction directly corresponds to their length.")
        st.markdown("* Base value is the value that would be predicted if we did not know any features for the current output.")
    st.divider()
    st.markdown("### Model Accuracy: ")
    "The model is trained on Random Forest and has had preprocessing done, on Orange Platform"
    st.markdown("> AUC score - `0.772`")
    st.markdown("> Classification Accuracy - `0.791`")