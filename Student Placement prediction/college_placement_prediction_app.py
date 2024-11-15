import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\collegePlace.csv")

# Preprocess the data
le_stream = LabelEncoder()
le_gender = LabelEncoder()
df['Stream'] = le_stream.fit_transform(df['Stream'])
df['Gender'] = le_gender.fit_transform(df['Gender'])

# Prepare the data for training
x = df.drop(columns=['PlacedOrNot', 'Age', 'Hostel'])
y = df['PlacedOrNot']

# Train a Decision Tree classifier
rf = DecisionTreeClassifier(max_depth=6, random_state=3)
rf.fit(x, y)

# Function to predict placement
def predict_place(gender, str1, interns, cgpa, backlog):
    data = [[gender, str1, interns, cgpa, backlog]]
    prediction = rf.predict(data)
    return prediction

def main():
    st.title("College Placement Prediction")
    st.write("Please enter the following details:")

    # Select Stream
    stream_options = le_stream.classes_
    str1 = st.selectbox("Select Stream:", stream_options)

    # Enter CGPA, Internships, Backlogs
    cgpa = st.number_input("Enter CGPA:")
    interns = st.number_input("Enter previous internships:")
    backlog = st.number_input("Enter backlogs:")

    # Select Gender
    gender_options = le_gender.classes_
    gender = st.selectbox("Select Gender:", gender_options)

    if st.button('Predict'):
        str1_encoded = le_stream.transform([str1])[0]
        gender_encoded = le_gender.transform([gender])[0]
        prediction = predict_place(gender_encoded, str1_encoded, interns, cgpa, backlog)
        if prediction == 1:
            st.success("The student is predicted to be placed.")
        else:
            st.warning("The student is predicted not to be placed.")

if __name__ == '__main__':
    main()
