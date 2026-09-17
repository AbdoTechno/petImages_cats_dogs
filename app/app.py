import streamlit as st 
import tensorflow as tf
import numpy as np
from helpers import load_model, load_and_preprocess_image, get_prediction



st.set_page_config(page_title="Cat vs Dog", page_icon="🐈", layout="centered")

st.title("Cat vs Dog")