import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import pickle

# Load the trained model
lg = pickle.load(open('placement.pkl', 'rb'))

# Web app title and image
img = Image.open('Job-Placement-Agency.jpg')
st.image(img, width=650)
st.title("Job Placement Prediction Model")

# Input features from user
input_text = st.text_input("Enter all features separated by commas (,)")
if input_text:
    try:
        # Convert input string to array of floats
        input_list = [float(x.strip()) for x in input_text.split(',')]
        np_df = np.asarray(input_list).reshape(1, -1)

        # Make prediction
        prediction = lg.predict(np_df)

        # Display prediction result
        if prediction[0] == 1:
            st.write("This Person Is Placed")
        else:
            st.write("This Person is not Placed")

    except ValueError:
        st.error("Please enter valid numeric values separated by commas.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
