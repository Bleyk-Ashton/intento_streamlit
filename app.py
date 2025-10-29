import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import cv2
#intento 
# --- Cargar modelo ---

modelo_cnn = load_model("model.keras")

st.title("Clasificación Male/Female con Grad-CAM y Saliency Map")

# --- Subida de imagen ---
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Leer imagen con OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img, caption="Imagen cargada", use_column_width=True)

    # --- Preprocesamiento ---
    img_resized = cv2.resize(img, (224, 224))  # Ajustar tamaño según tu modelo
    img_input = img_resized / 255.0             # Normalizar
    img_input = np.expand_dims(img_input, axis=0)

    # --- Predicción ---
    pred = modelo_cnn.predict(img_input)
    prob_male = float(pred[0,0])
    prob_female = 1 - prob_male
    pred_class = "Male" if prob_male > 0.5 else "Female"

    st.write(f"**Predicción:** {pred_class}")
    st.write(f"**Probabilidad Female:** {prob_female:.2f}")
    st.write(f"**Probabilidad Male:** {prob_male:.2f}")

    # --- Función Grad-CAM ---
    def grad_cam_sequential(model, image_tensor, class_index, target_layer_index):
        with tf.GradientTape() as tape:
            x = image_tensor
            for i, layer in enumerate(model.layers):
                x = layer(x)
                if i == target_layer_index:
                    conv_outputs = x
            predictions = x
            loss = predictions[:,0] if class_index == 1 else 1 - predictions[:,0]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        heatmap /= tf.reduce_max(heatmap) + 1e-8
        return heatmap.numpy()

    # --- Función Saliency Map ---
    def saliency_map(model, image_tensor, class_index):
        image_tensor = tf.convert_to_tensor(image_tensor)
        image_tensor = tf.Variable(image_tensor)
        with tf.GradientTape() as tape:
            tape.watch(image_tensor)
            pred = model(image_tensor)
            loss = pred[:,0] if class_index == 1 else 1 - pred[:,0]
        grads = tape.gradient(loss, image_tensor)[0]
        saliency = tf.reduce_max(tf.abs(grads), axis=-1)
        saliency = (saliency - tf.reduce_min(saliency)) / (tf.reduce_max(saliency) - tf.reduce_min(saliency) + 1e-8)
        return saliency.numpy()

    # --- Visualización Grad-CAM ---
    target_layers_idx = [0, 2, 4]  # conv1_block1, conv2_block2, conv3_block3

    st.subheader("Grad-CAM")
    for idx_layer in target_layers_idx:
        heatmap = grad_cam_sequential(modelo_cnn, img_input, class_index=1 if pred_class=="Male" else 0, target_layer_index=idx_layer)
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap_resized = np.uint8(255 * heatmap_resized)
        superimposed_img = cv2.addWeighted(img, 0.6, cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET), 0.4, 0)
        st.image(superimposed_img, caption=f"Grad-CAM: {modelo_cnn.layers[idx_layer].name}", use_column_width=True)

    # --- Visualización Saliency Map ---
    st.subheader("Saliency Map")
    sal_map = saliency_map(modelo_cnn, img_input, class_index=1 if pred_class=="Male" else 0)
    sal_map_resized = cv2.resize(sal_map, (img.shape[1], img.shape[0]))
    sal_map_img = np.uint8(255 * sal_map_resized)
    sal_map_img = cv2.applyColorMap(sal_map_img, cv2.COLORMAP_JET)
    superimposed_sal = cv2.addWeighted(img, 0.6, sal_map_img, 0.4, 0)
    st.image(superimposed_sal, caption="Saliency Map", use_column_width=True)
