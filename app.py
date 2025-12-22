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
NOM_DERNIERE_COUCHE_CONV = "TumeurD2"  # Vérifiez ce nom dans votre modèle
NOMS_CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
CHEMIN_MODELE = "modele.h5"  # Changez ce chemin si nécessaire

# FONCTIONS DE CHARGEMENT ET PRÉTRAITEMENT

@st.cache_resource
def charger_modele():
    """
    Charge le modèle Keras pré-entraîné depuis le disque.
    """
    try:
        # Essayer plusieurs chemins possibles
        chemins_possibles = [
            CHEMIN_MODELE,
            "modele.h5",
            "model.h5",
            "modele/modele.h5",
            "/mount/src/your-repo-name/modele.h5"  # Chemin sur Streamlit Cloud
        ]
        
        chemin_trouve = None
        for chemin in chemins_possibles:
            if os.path.exists(chemin):
                chemin_trouve = chemin
                break
        
        if not chemin_trouve:
            st.error("Fichier modèle introuvable. Chemins essayés:")
            for chemin in chemins_possibles:
                st.write(f"- {chemin}")
            st.stop()
        
        modele = keras.models.load_model(chemin_trouve, compile=False)
        st.sidebar.success(f"Modèle chargé: {chemin_trouve}")
        return modele
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {str(e)}")
        st.stop()

def preparer_image(image_pil, taille=TAILLE_IMAGE):
    """
    Prétraite l'image pour la classification.
    """
    image = image_pil.resize(taille)
    image_array = keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array / 255.0
    return image_array

# FONCTIONS GRAD-CAM - CORRIGÉES

def make_gradcam_heatmap(model, img_array, layer_name, pred_index=None):
    """
    Version simplifiée et corrigée de Grad-CAM.
    Évite les erreurs d'indexation par tuple.
    """
    try:
        # Créer un modèle qui retourne les activations de la couche et les prédictions
        grad_model = tf.keras.models.Model(
            [model.inputs], 
            [model.get_layer(layer_name).output, model.output]
        )
        
        # Enregistrer les opérations sous la bande de gradient
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            
            # Extraire la prédiction pour la classe cible
            # CORRECTION: Utiliser [0, pred_index] au lieu de [:, pred_index]
            class_channel = predictions[0, pred_index]
        
        # Extraire les gradients
        grads = tape.gradient(class_channel, conv_outputs)
        
        # CORRECTION: Vérifier que grads n'est pas None
        if grads is None:
            # Fallback: utiliser les activations moyennes
            heatmap = tf.reduce_mean(conv_outputs[0], axis=-1)
            return heatmap.numpy(), predictions.numpy()
        
        # Pooling des gradients sur les axes spatiaux
        # CORRECTION: Utiliser axis=[0, 1, 2] au lieu de axis=(0, 1, 2)
        pooled_grads = tf.reduce_mean(grads, axis=[0, 1, 2])
        
        # Multiplier chaque canal par le gradient moyen correspondant
        conv_outputs = conv_outputs[0]  # Shape: (height, width, channels)
        
        # CORRECTION: Éviter l'opérateur @ qui peut causer des problèmes
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        
        # ReLU et normalisation
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.reduce_max(heatmap)
        
        if max_val > 0:
            heatmap = heatmap / max_val
        else:
            # Si tout est zéro, créer une heatmap uniforme
            heatmap = tf.zeros_like(heatmap)
        
        return heatmap.numpy(), predictions.numpy()
    
    except Exception as e:
        st.error(f"Erreur Grad-CAM détaillée: {str(e)}")
        
        # Debug: Afficher des informations supplémentaires
        try:
            st.write("Debug - Informations sur le modèle:")
            st.write(f"Nombre de couches: {len(model.layers)}")
            st.write(f"Couches disponibles (5 dernières):")
            for layer in model.layers[-5:]:
                st.write(f"  - {layer.name} ({layer.__class__.__name__})")
        except:
            pass
        
        return None, None

def superposer_gradcam(img_array, heatmap, alpha=0.4):
    """
    Superpose la carte Grad-CAM sur l'image.
    """
    try:
        if heatmap is None:
            return Image.fromarray(img_array.astype(np.uint8))
        
        # Redimensionner la heatmap
        heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
        
        # Normaliser entre 0 et 1
        if heatmap_resized.max() > 0:
            heatmap_resized = heatmap_resized / heatmap_resized.max()
        
        # Appliquer la colormap
        colormap = mpl.colormaps["jet"]
        heatmap_colored = colormap(heatmap_resized)[:, :, :3]  # Ignorer alpha
        
        # Convertir en 0-255
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # Superposer
        superimposed = heatmap_colored * alpha + img_array * (1 - alpha)
        superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
        
        return Image.fromarray(superimposed)
    except Exception as e:
        st.error(f"Erreur superposition: {str(e)}")
        return Image.fromarray(img_array.astype(np.uint8))

# FONCTIONS D'AFFICHAGE

def afficher_predictions_detailees(predictions):
    """
    Affiche les probabilités de prédiction.
    """
    st.subheader("Probabilités détaillées")
    
    cols = st.columns(4)
    for i, classe in enumerate(NOMS_CLASSES):
        proba = predictions[i] * 100
        with cols[i]:
            st.markdown(f"**{classe.capitalize()}**")
            st.progress(float(proba/100), text=f"{proba:.1f}%")

def debug_model_layers(model):
    """
    Fonction de debug pour afficher les couches du modèle.
    """
    with st.sidebar.expander("Debug - Couches du modèle"):
        st.write(f"Total couches: {len(model.layers)}")
        st.write("Noms des couches (recherche de couches convolutionnelles):")
        
        conv_layers = []
        for i, layer in enumerate(model.layers):
            layer_type = layer.__class__.__name__
            if 'conv' in layer_type.lower() or 'Conv' in layer.name:
                conv_layers.append((i, layer.name, layer_type))
        
        if conv_layers:
            st.write("Couches convolutionnelles trouvées:")
            for i, name, ltype in conv_layers:
                st.write(f"  {i}: {name} ({ltype})")
        else:
            st.write("Aucune couche convolutionnelle trouvée avec 'conv' dans le nom")

# APPLICATION PRINCIPALE

def main():
    """
    Fonction principale de l'application.
    """
    # Configuration
    st.set_page_config(
        page_title="Analyse d'IRM Cérébrales",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Système d'Analyse d'IRM Cérébrales")
    st.markdown("Classification automatique avec explication visuelle par Grad-CAM")
    
    # Sidebar
    st.sidebar.header("Paramètres")
    
    # Option pour choisir la couche manuellement
    st.sidebar.subheader("Configuration Grad-CAM")
    
    # Charger le modèle d'abord
    model = charger_modele()
    
    # Debug optionnel
    if st.sidebar.checkbox("Activer le mode debug"):
        debug_model_layers(model)
    
    # Paramètres
    alpha = st.sidebar.slider(
        "Transparence Grad-CAM", 0.1, 0.8, 0.4, 0.1
    )
    
    # Téléchargement d'image
    st.header("Téléchargement d'image")
    
    uploaded_file = st.file_uploader(
        "Choisissez une image IRM",
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        try:
            # Ouvrir l'image
            image = Image.open(uploaded_file).convert("RGB")
            original_image = image.copy()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Image originale")
                st.image(image, use_container_width=True)
            
            with col2:
                st.subheader("Zone d'analyse")
                
                if st.button("🔍 Lancer l'analyse", type="primary", use_container_width=True):
                    with st.spinner("Analyse en cours..."):
                        # Préparer l'image
                        img_array = preparer_image(image)
                        
                        # Prédiction
                        predictions = model.predict(img_array, verbose=0)[0]
                        pred_index = np.argmax(predictions)
                        predicted_class = NOMS_CLASSES[pred_index]
                        confidence = predictions[pred_index] * 100
                        
                        # Afficher résultats
                        st.success(f"**Résultat: {predicted_class.capitalize()}** (confiance: {confidence:.1f}%)")
                        
                        # Générer Grad-CAM
                        heatmap, _ = make_gradcam_heatmap(
                            model, 
                            img_array, 
                            NOM_DERNIERE_COUCHE_CONV,
                            pred_index
                        )
                        
                        # Fallback si la couche spécifiée ne fonctionne pas
                        if heatmap is None:
                            st.warning(f"La couche '{NOM_DERNIERE_COUCHE_CONV}' ne fonctionne pas. Recherche d'une couche alternative...")
                            
                            # Essayer avec différentes couches
                            alternative_layers = []
                            for layer in model.layers:
                                if 'conv' in layer.name.lower() or 'activation' in layer.name:
                                    alternative_layers.append(layer.name)
                            
                            if alternative_layers:
                                for layer_name in alternative_layers[:3]:  # Essayer 3 premières
                                    st.write(f"Essai avec la couche: {layer_name}")
                                    heatmap, _ = make_gradcam_heatmap(
                                        model, img_array, layer_name, pred_index
                                    )
                                    if heatmap is not None:
                                        NOM_DERNIERE_COUCHE_CONV = layer_name
                                        st.info(f"Utilisation de la couche: {layer_name}")
                                        break
                        
                        if heatmap is not None:
                            # Préparer l'image pour superposition
                            img_for_overlay = np.array(original_image.resize(TAILLE_IMAGE))
                            
                            # Superposer Grad-CAM
                            superimposed_img = superposer_gradcam(
                                img_for_overlay, 
                                heatmap, 
                                alpha
                            )
                            
                            # Afficher le résultat
                            st.image(
                                superimposed_img,
                                caption=f"Visualisation Grad-CAM - {predicted_class}",
                                use_container_width=True
                            )
                            
                            # Détails des prédictions
                            afficher_predictions_detailees(predictions)
                            
                            # Explications
                            with st.expander("ℹ️ Interprétation"):
                                st.markdown("""
                                **Grad-CAM (Gradient-weighted Class Activation Mapping):**
                                - Zones **rouges/jaunes**: Régions déterminantes pour la décision
                                - Zones **bleues**: Régions moins importantes
                                
                                **Note médicale:** Cet outil est une aide au diagnostic.
                                Consultez toujours un professionnel de santé.
                                """)
                        else:
                            st.error("Impossible de générer la visualisation Grad-CAM.")
                            st.info("Affichage des prédictions uniquement:")
                            afficher_predictions_detailees(predictions)
        
        except Exception as e:
            st.error(f"Erreur de traitement: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    else:
        st.info(" Veuillez télécharger une image IRM pour commencer l'analyse")
        
        # Section exemple
        with st.expander("Comment utiliser cette application"):
            st.markdown("""
            1. **Téléchargez** une image IRM cérébrale
            2. **Cliquez** sur "Lancer l'analyse"
            3. **Visualisez** le diagnostic et les explications
            
            **Formats acceptés:** JPG, JPEG, PNG
            **Résolution recommandée:** 224x224 pixels
            """)
    
    # Pied de page
    st.divider()
    st.caption("Application d'aide au diagnostic - À utiliser avec l'expertise médicale appropriée")

if __name__ == "__main__":
    main()
