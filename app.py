import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import os
import cv2

# --- CONFIGURATION DES CONSTANTES ---
TAILLE_CLASSIF = (224, 224)
TAILLE_SEGMENT = (128, 128)
NOMS_CLASSES = ['Glioma', 'Méningiome', 'Pas de tumeur', 'Hypophyse']
CHEMIN_CLF = "modele.h5"
CHEMIN_SEG = "segmentation.h5"

# --- FONCTIONS TECHNIQUES ---

def dice_coef(y_true, y_pred, smooth=100):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

@st.cache_resource
def charger_modeles():
    """Charge les deux modèles IA simultanément."""
    clf = keras.models.load_model(CHEMIN_CLF, compile=False)
    # Chargement du modèle de segmentation avec la fonction personnalisée Dice
    seg = keras.models.load_model(CHEMIN_SEG, custom_objects={'dice_coef': dice_coef}, compile=False)
    return clf, seg

def segmenter_tumeur(image_pil, modele_seg):
    """Applique le masque de segmentation et calcule la surface."""
    # Préparation
    img = image_pil.resize(TAILLE_SEGMENT)
    img_array = np.array(img.convert('RGB')) / 255.0
    img_input = np.expand_dims(img_array, axis=0)
    
    # Prédiction
    pred = modele_seg.predict(img_input, verbose=0)[0]
    mask = (pred > 0.5).astype(np.uint8)
    
    # Calcul volumétrique
    pixels_tumeur = np.sum(mask)
    pourcentage = (pixels_tumeur / (128 * 128)) * 100
    
    # Création de l'Overlay (Superposition rouge)
    mask_visual = np.zeros_like(img_array)
    mask_visual[:,:,0] = mask.squeeze() * 1.0  # Canal Rouge
    
    overlay = cv2.addWeighted(img_array.astype(np.float32), 0.7, mask_visual.astype(np.float32), 0.3, 0)
    return overlay, pixels_tumeur, pourcentage

# --- INTERFACE UTILISATEUR ---

def main():
    st.set_page_config(page_title="NeuroVision AI", page_icon="🧠", layout="wide")
    
    # CSS pour un look "Médical futuriste"
    st.markdown("""
        <style>
        .report-box { background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #3498db; }
        .metric-text { font-size: 24px; font-weight: bold; color: #e74c3c; }
        </style>
    """, unsafe_allow_html=True)

    st.title("🧠 NeuroVision AI : Diagnostic & Segmentation")
    
    # Chargement des modèles
    with st.sidebar:
        st.header("État du Système")
        clf_model, seg_model = charger_modeles()
        st.success("Modèles Classification & Segmentation Chargés")

    uploaded_file = st.file_uploader("Charger une IRM cérébrale...", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.subheader("🖼️ Image Originale")
            st.image(image, use_container_width=True)

        if st.button("Lancer l'Analyse Complète", type="primary"):
            # 1. CLASSIFICATION
            img_clf = np.array(image.resize(TAILLE_CLASSIF)) / 255.0
            pred_clf = clf_model.predict(np.expand_dims(img_clf, axis=0), verbose=0)[0]
            idx = np.argmax(pred_clf)
            classe = NOMS_CLASSES[idx]
            confiance = pred_clf[idx] * 100

            # 2. SEGMENTATION
            overlay, px, percent = segmenter_tumeur(image, seg_model)

            with col2:
                st.subheader("🎯 Segmentation IA")
                st.image(overlay, use_container_width=True, caption="Zone tumorale détectée (Rouge)")

            with col3:
                st.subheader("📊 Rapport d'Analyse")
                st.markdown(f"""
                <div class="report-box">
                    <b>Type détecté :</b> {classe}<br>
                    <b>Confiance :</b> {confiance:.1f}%<br><br>
                    <b>Analyse Spatiale :</b><br>
                    <span class="metric-text">{percent:.2f}%</span> de la zone cérébrale affectée<br>
                    ({px} pixels segmentés)
                </div>
                """, unsafe_allow_html=True)

                if percent > 0.1:
                    st.error("⚠️ Présence de masse tumorale confirmée par segmentation.")
                else:
                    st.success("✅ Aucune masse significative détectée par segmentation.")

if __name__ == "__main__":
    main()
