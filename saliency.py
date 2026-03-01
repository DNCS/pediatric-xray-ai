import tensorflow as tf
import numpy as np
import cv2
import os

def generate_saliency(model, img_path, output_path, img_size=(224,224)):

    # Load image
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
    img_arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = tf.convert_to_tensor(img_arr)

    # Gradient wrt input
    with tf.GradientTape() as tape:
        tape.watch(img_arr)
        preds = model(img_arr)
        pred_idx = tf.argmax(preds[0])
        loss = preds[:, pred_idx]

    grads = tape.gradient(loss, img_arr)[0]
    saliency = tf.reduce_max(tf.abs(grads), axis=-1).numpy()

    # Normalize
    saliency = (saliency - saliency.min()) / (saliency.max() + 1e-8)

    saliency = cv2.resize(saliency, img_size)
    saliency = np.uint8(255 * saliency)
    saliency = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)

    original = cv2.imread(img_path)
    original = cv2.resize(original, img_size)

    overlay = cv2.addWeighted(original, 0.6, saliency, 0.4, 0)

    cv2.imwrite(output_path, overlay)

    return output_path