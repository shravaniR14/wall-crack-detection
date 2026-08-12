import streamlit as st
import numpy as np
from PIL import Image
import joblib
from skimage.feature import hog

model = joblib.load("hog_crack_model.pkl")

st.title("Wall Crack Detection")

st.write("Upload a wall image to check for cracks.")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image")

    image = image.resize((32, 32))
    image = image.convert("L")

    image = np.array(image)

    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2)
    )

    features = features.reshape(1, -1)

    prediction = model.predict(features)
    probability = model.predict_proba(features)

    if prediction[0] == 1:
        confidence = probability[0][1] * 100
        st.error("Crack Detected")
        st.write("Confidence:", round(confidence, 2), "%")

    else:
        confidence = probability[0][0] * 100
        st.success("No Crack Detected")
        st.write("Confidence:", round(confidence, 2), "%")

        uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"]
)