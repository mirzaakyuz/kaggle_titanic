import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier

from PIL import Image

st.markdown('## Titanic Survival Prediction App')

img = Image.open("survive-unsplash.jpg")
caption = """
Photo by [Li Yang] (https://unsplash.com/@ly0ns?) on [Unsplash] (https://unsplash.com/s/photos/survive?)"""
st.image(img, width=400)
st.write(caption)

st.write("""

            This app predicts whether the passanger survived or not using the given features.

            Data obtained from [Kaggle] (https://www.kaggle.com/c/titanic/)

""")
st.sidebar.header("User Input Features")

st.sidebar.markdown("""
[Example CSV Input File](https://drive.google.com/file/d/1AwdPIHRFE1RAwXBqSmmD_WbvXzI09NC0/view?usp=sharing)
""")

uploaded_file = st.sidebar.file_uploader("Upload your input file", type=["csv"])

if uploaded_file is not None:
    df_for_predict = pd.read_csv(uploaded_file)
    st.subheader('Your entry:')
    st.write(df_for_predict)
    load_clf = pickle.load(open('titanic_rf_clf.pkl', 'rb'))

    prediction = load_clf.predict(df_for_predict)
    prediction_proba = load_clf.predict_proba(df_for_predict)

    st.subheader('Prediction')
    result = np.array(['Not Survived', 'Survived'])
    st.write(result[prediction])

    st.subheader('Prediction Probability')
    st.write('0 for Not-Survived,1 for Survived')
    st.write(prediction_proba)

else:
    def user_input_fetures():
        Age = st.sidebar.slider('Passenger Age', 0,80,28)
        SibSp = st.sidebar.slider('Number of siblings/spouse aboard Titanic',0,8,1)
        Parch = st.sidebar.slider('Number of parents/children aboard Titanic',0,6,1)
        Fare = st.sidebar.slider('Passenger Fare',0,513,40)
        Sex = st.sidebar.selectbox("Gender of Passenger",('female','male'))
        Embarked = st.sidebar.selectbox("Where the Passenger embarked?",('Cherbourg', 'Queenstown', 'Southampton'))
        is_group = st.sidebar.selectbox("Has the Passanger group ticket? 0-No 1-Yes",(0,1))
        Pclass = st.sidebar.selectbox("Passneger Class", (1,2,3))


        data = { 'Age' : Age,
                 'SibSp': SibSp,
                 'Parch': Parch,
                 'Fare': Fare,
                 'Sex' : Sex,
                 'Embarked': Embarked,
                 'Pclass': Pclass,
                 'is_group': is_group
                 }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_fetures()

    st.subheader('Your entry:')
    st.write(input_df)
    input_df['Embarked'] = input_df['Embarked'].str[0]

    df_train_raw = pd.read_csv('df_encode.csv')
    df = pd.concat([df_train_raw, input_df], axis=0)


    encode = ['Sex', 'Embarked', 'Pclass']
    for col in encode:
        dummy = pd.get_dummies(df[col], prefix=col ,drop_first=True)
        df = pd.concat([df,dummy], axis=1)
        del df[col]



    df["is_not_alone"] = ((df["SibSp"] != 0) | (df["Parch"] != 0)) * 1

    df['Sex_female'] = df['Sex_male'].map({0:1,1:0})
    df.drop('Sex_male',axis=1,inplace=True)

    df = df[['Age', 'SibSp', 'Parch', 'Fare', 'is_group', 'is_not_alone', 'Sex_female', 'Embarked_Q', 'Embarked_S', 'Pclass_2', 'Pclass_3']]


    df_for_predict = df.iloc[-1].values.reshape(1,11)


    load_clf = pickle.load(open('titanic_rf_clf.pkl', 'rb'))

    prediction = load_clf.predict(df_for_predict)
    prediction_proba = load_clf.predict_proba(df_for_predict)

    st.subheader('Prediction')
    result = np.array(['Not Survived', 'Survived'])
    st.write(result[prediction])

    st.subheader('Prediction Probability')
    st.write('0 for Not-Survived, 1 for Survived')
    st.write(prediction_proba)
