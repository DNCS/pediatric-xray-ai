import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

#from gradcam import generate_gradcam
from saliency import generate_saliency
from save_predictions import save_prediction_to_db

MODEL_PATH = "models/best_model_fixed.keras"
GRADCAM_DIR = "outputs/gradcam"
os.makedirs(GRADCAM_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
CLASS_LABELS = {0: "Good X-ray", 1: "Bad X-ray"}

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()


# --------------------------
# FIND LAST CONV LAYER
# --------------------------
def find_last_conv(model):
    backbone = model.get_layer("functional")

    last_conv = None
    for layer in reversed(backbone.layers):
        try:
            if len(layer.output.shape) == 4:
                last_conv = layer
                break
        except:
            continue

    if last_conv is None:
        st.error("No convolution layer found!")
        st.stop()

    return last_conv.name


st.title("🩻 Pediatric X-Ray Quality Checker")
uploaded = st.file_uploader("Upload X-Ray", type=["jpg","jpeg","png"])

if uploaded:

    st.image(uploaded, use_column_width=True)

    temp_path = "temp_image.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    img = image.load_img(temp_path, target_size=IMG_SIZE)
    img_arr = image.img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    preds = model.predict(img_arr)
    pred_idx = int(np.argmax(preds[0]))
    pred_label = CLASS_LABELS[pred_idx]

    st.subheader("Prediction")
    st.write(f"### {pred_label}")

    for i, p in enumerate(preds[0]):
        st.write(f"{CLASS_LABELS[i]}: {p:.4f}")

    # Grad-CAM layer
   #### last_conv = find_last_conv(model)
   #### st.write(f"Grad-CAM layer: **{last_conv}**")

    # Generate heatmap
   #### grad_path = os.path.join(GRADCAM_DIR, f"grad_{uploaded.name}")

   #### generate_gradcam(
   ####     model=model,
   ####     img_path=temp_path,
   ####     output_path=grad_path,
   ####     layer_name=last_conv
   #### )

   #### st.subheader("Grad-CAM Heatmap")
   #### st.image(grad_path, use_column_width=True)
    st.subheader("Model Attention Map")

    saliency_path = os.path.join(GRADCAM_DIR, f"saliency_{uploaded.name}")

    generate_saliency(
        model=model,
        img_path=temp_path,
        output_path=saliency_path,
        img_size=IMG_SIZE
    )

    st.image(saliency_path, caption="Attention Map", use_column_width=True)
    if st.button("Save to DB"):
        save_prediction_to_db(
            img_path=temp_path,
            probabilities={CLASS_LABELS[i]: float(preds[0][i]) for i in CLASS_LABELS},
            prediction_label=pred_label,
            gradcam_image_path=grad_path
        )
        st.success("Saved to database!")
