"""
Application de classification d'IRM cérébrales avec explication par Grad-CAM
Application professionnelle pour l'analyse assistée d'images médicales
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
from PIL import Image
import os

# CONFIGURATION DES CONSTANTES

TAILLE_IMAGE = (224, 224)
NOM_DERNIERE_COUCHE_CONV = "TumeurD2"
NOMS_CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
CHEMIN_MODELE = "modele/modele.h5"

# FONCTIONS DE CHARGEMENT ET PRÉTRAITEMENT

@st.cache_resource
def charger_modele():
    """
    Charge le modèle Keras pré-entraîné depuis le disque.
    
    Returns:
        model: Modèle Keras chargé
    """
    try:
        if not os.path.exists(CHEMIN_MODELE):
            st.error(f"Fichier modèle introuvable : {CHEMIN_MODELE}")
            st.info("Veuillez placer le modèle dans le répertoire 'modele/'")
            st.stop()
        
        modele = keras.models.load_model(CHEMIN_MODELE)
        st.sidebar.success("Modèle chargé avec succès")
        return modele
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {str(e)}")
        st.stop()

def preparer_image(image_pil, taille=TAILLE_IMAGE):
    """
    Prétraite l'image pour la classification par le modèle.
    
    Args:
        image_pil: Image PIL à prétraiter
        taille: Dimensions cibles de l'image
    
    Returns:
        numpy.ndarray: Image prétraitée sous forme de tableau numpy
    """
    image = image_pil.resize(taille)
    image_array = keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# FONCTIONS GRAD-CAM

def creer_gradcam(model, img_array, layer_name, pred_index=None):
    """
    Génère la carte d'activation Grad-CAM pour visualiser les régions importantes.
    
    Args:
        model: Modèle Keras
        img_array: Image d'entrée sous forme de tableau numpy
        layer_name: Nom de la couche convolutive à utiliser
        pred_index: Index de la classe à expliquer (None = classe prédite)
    
    Returns:
        tuple: (heatmap, predictions) ou (None, None) en cas d'erreur
    """
    try:
        grad_model = keras.models.Model(
            [model.inputs], 
            [model.get_layer(layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            
            class_channel = predictions[:, pred_index]
        
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy(), predictions.numpy()
    
    except Exception as e:
        st.error(f"Erreur lors de la génération Grad-CAM : {str(e)}")
        return None, None

def superposer_gradcam(img_array, heatmap, alpha=0.4):
    """
    Superpose la carte Grad-CAM sur l'image originale.
    
    Args:
        img_array: Image originale sous forme de tableau numpy
        heatmap: Carte de chaleur Grad-CAM
        alpha: Transparence de la superposition
    
    Returns:
        PIL.Image: Image avec superposition Grad-CAM
    """
    heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    
    colormap = mpl.colormaps["jet"]
    colors = colormap(np.arange(256))[:, :3]
    colored_heatmap = colors[heatmap]
    
    colored_heatmap = keras.utils.array_to_img(colored_heatmap)
    colored_heatmap = keras.utils.img_to_array(colored_heatmap)
    
    superimposed_img = colored_heatmap * alpha + img_array
    superimposed_img = keras.utils.array_to_img(superimposed_img)
    
    return superimposed_img

# FONCTIONS D'AFFICHAGE ET D'INTERFACE

def afficher_predictions_detailees(predictions):
    """
    Affiche les probabilités de prédiction sous forme de barres de progression.
    
    Args:
        predictions: Tableau de probabilités pour chaque classe
    """
    st.subheader("Probabilités détaillées par classe")
    
    cols = st.columns(4)
    for i, classe in enumerate(NOMS_CLASSES):
        proba = predictions[i] * 100
        with cols[i]:
            # Détermination de la couleur en fonction du seuil
            if proba > 70:
                color = "green"
            elif proba > 30:
                color = "orange"
            else:
                color = "gray"
            
            st.markdown(f"<h4>{classe.capitalize()}</h4>", unsafe_allow_html=True)
            st.progress(
                min(int(proba), 100) / 100,
                text=f"{proba:.1f}%"
            )

def afficher_informations_modele():
    """
    Affiche les informations techniques du modèle dans la sidebar.
    """
    with st.sidebar.expander("Informations techniques"):
        st.write(f"**Couche Grad-CAM:** {NOM_DERNIERE_COUCHE_CONV}")
        st.write(f"**Taille d'entrée:** {TAILLE_IMAGE}")
        st.write(f"**Architecture:** CNN avec couches convolutives")
        st.write(f"**Classes:** {', '.join(NOMS_CLASSES)}")

def afficher_guide_utilisation():
    """
    Affiche le guide d'utilisation dans la sidebar.
    """
    with st.sidebar.expander("Guide d'utilisation"):
        st.markdown("""
        **Procédure:**
        1. Téléchargez une image IRM cérébrale
        2. Cliquez sur 'Analyser avec Grad-CAM'
        3. Visualisez la prédiction et les explications
        
        **Classes diagnostiques:**
        - **Glioma**: Tumeur du tissu glial cérébral
        - **Meningioma**: Tumeur des méninges
        - **Notumor**: Absence de tumeur détectée
        - **Pituitary**: Tumeur de l'hypophyse
        
        **Format supporté:** JPG, JPEG, PNG
        """)

# APPLICATION PRINCIPALE

def main():
    """
    Fonction principale de l'application Streamlit.
    """
    # Configuration de la page
    st.set_page_config(
        page_title="Système d'Analyse d'IRM Cérébrales",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # En-tête de l'application
    st.title("Système d'Analyse d'IRM Cérébrales")
    st.markdown("""
    Application d'aide au diagnostic par intelligence artificielle.
    Classification automatique des images IRM avec explication des prédictions.
    """)
    
    st.divider()
    
    # Sidebar - Paramètres
    st.sidebar.header("Paramètres d'analyse")
    
    # Paramètre de transparence Grad-CAM
    alpha = st.sidebar.slider(
        "Intensité de visualisation Grad-CAM",
        min_value=0.1,
        max_value=0.8,
        value=0.4,
        step=0.1,
        help="Contrôle la visibilité de la superposition"
    )
    
    # Chargement du modèle
    st.sidebar.subheader("Modèle")
    model = charger_modele()
    
    # Informations techniques
    afficher_informations_modele()
    
    # Guide d'utilisation
    afficher_guide_utilisation()
    
    # Section principale - Téléchargement d'image
    st.header("Analyse d'image IRM")
    
    uploaded_file = st.file_uploader(
        "Sélectionnez une image IRM cérébrale",
        type=["jpg", "jpeg", "png"],
        help="Formats supportés: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        # Affichage de l'image originale
        try:
            image = Image.open(uploaded_file).convert("RGB")
            original_image = image.copy()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Image originale")
                st.image(image, use_container_width=True)
            
            # Bouton d'analyse
            if st.button("Lancer l'analyse Grad-CAM", type="primary"):
                with st.spinner("Analyse en cours..."):
                    # Préparation de l'image
                    img_array = preparer_image(image)
                    
                    # Prédiction
                    predictions = model.predict(img_array, verbose=0)[0]
                    pred_index = np.argmax(predictions)
                    predicted_class = NOMS_CLASSES[pred_index]
                    confidence = predictions[pred_index] * 100
                    
                    # Affichage des résultats
                    st.success(f"Diagnostic prédit : **{predicted_class.capitalize()}**")
                    
                    col_metric1, col_metric2 = st.columns(2)
                    with col_metric1:
                        st.metric("Confiance", f"{confidence:.1f}%")
                    with col_metric2:
                        st.metric("Classe", predicted_class.capitalize())
                    
                    # Génération Grad-CAM
                    heatmap, _ = creer_gradcam(
                        model, 
                        img_array, 
                        NOM_DERNIERE_COUCHE_CONV,
                        pred_index
                    )
                    
                    if heatmap is not None:
                        # Préparation pour superposition
                        img_for_overlay = keras.utils.img_to_array(
                            original_image.resize(TAILLE_IMAGE)
                        ).astype(np.uint8)
                        
                        # Superposition Grad-CAM
                        superimposed_img = superposer_gradcam(
                            img_for_overlay, 
                            heatmap, 
                            alpha
                        )
                        
                        # Affichage Grad-CAM
                        with col2:
                            st.subheader("Visualisation Grad-CAM")
                            st.image(
                                superimposed_img, 
                                caption=f"Régions déterminantes pour '{predicted_class}'",
                                use_container_width=True
                            )
                        
                        # Détails des prédictions
                        afficher_predictions_detailees(predictions)
                        
                        # Explication technique
                        with st.expander("Interprétation de la visualisation Grad-CAM"):
                            st.markdown("""
                            **Légende des couleurs:**
                            
                            - **Zones rouges/jaunes**: Régions les plus influentes dans la décision du modèle
                            - **Zones bleues**: Régions moins pertinentes pour la classification
                            
                            **Méthodologie:**
                            La technique Grad-CAM (Gradient-weighted Class Activation Mapping) 
                            utilise les gradients de la classe prédite par rapport aux caractéristiques 
                            de la dernière couche convolutive pour identifier les régions importantes.
                            
                            **Note importante:**
                            Cette application constitue un outil d'aide au diagnostic et ne remplace pas 
                            l'expertise d'un professionnel de santé qualifié.
                            """)
                    
                    else:
                        st.warning("La génération de l'explication visuelle a échoué.")
        
        except Exception as e:
            st.error(f"Erreur lors du traitement de l'image : {str(e)}")
    
    # Pied de page
    st.divider()
    st.caption("Application développée à des fins de recherche médicale - Outil d'aide au diagnostic")
    st.caption("Les résultats doivent être validés par un radiologue qualifié")

# POINT D'ENTRÉE

if __name__ == "__main__":
    main()