import streamlit as st

from helpers import load_model, get_prediction

st.set_page_config(
    page_title="Cat vs Dog Comparison", page_icon="🐱", layout="centered"
)

# Load Models
cnn_model = load_model("../models/cat_dog_model_cnn.tflite")
vgg_model = load_model("../models/cat_dog_model_vgg.tflite")

# UI
st.title("Cat vs Dog")

st.write("Upload an image and compare the predictions of the CNN and VGG16 models.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", width=300)
    st.divider()
    # Predictions
    cnn_prediction, cnn_confidence = get_prediction(
        cnn_model, uploaded_file, model_type="cnn"
    )
    vgg_prediction, vgg_confidence = get_prediction(
        vgg_model, uploaded_file, model_type="vgg"
    )
    # Results
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("CNN")
        st.write(f"Prediction: **{cnn_prediction}**")
        st.write(f"Confidence: **{cnn_confidence:.2%}**")
        if cnn_prediction == "Cat":
            st.info("🐱 Cat")
        else:
            st.info("🐶 Dog")
    with col2:
        st.subheader("VGG16")
        st.write(f"Prediction: **{vgg_prediction}**")
        st.write(f"Confidence: **{vgg_confidence:.2%}**")
        if vgg_prediction == "Cat":
            st.info("🐱 Cat")
        else:
            st.info("🐶 Dog")
    # Comparison
    st.divider()
    st.subheader("Comparison")
    if cnn_prediction == vgg_prediction:
        st.success(f"Both models predicted: **{cnn_prediction}**")
    else:
        st.warning("The models made different predictions.")
        st.write(f"CNN → **{cnn_prediction}** ({cnn_confidence:.2%})")
        st.write(f"VGG16 → **{vgg_prediction}** ({vgg_confidence:.2%})")
else:
    st.warning("Please upload an image.")
