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
CHEMIN_MODELE = "modele.h5"

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
            "/mount/src/detection_tumeur/modele.h5"
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

# FONCTION GRAD-CAM CORRIGÉE - Version simplifiée

def make_gradcam_heatmap_simple(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Version simplifiée de Grad-CAM qui évite les erreurs d'indexation.
    Basée sur la documentation officielle de Keras.
    """
    try:
        # Créer un modèle qui mappe l'image d'entrée aux activations de la dernière couche conv
        grad_model = keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        # Calculer le gradient de la classe prédite par rapport aux activations
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            
            # CORRECTION IMPORTANTE: Ne pas utiliser d'indexation par tuple
            # Extraire la valeur scalaire de la classe prédite
            class_output = predictions[0][pred_index]
        
        # Calculer les gradients
        grads = tape.gradient(class_output, conv_outputs)[0]
        
        # Calculer les gradients pondérés
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Multiplier chaque canal par son gradient moyen
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        
        # Appliquer ReLU et normaliser
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / tf.reduce_max(heatmap)
        
        return heatmap.numpy(), predictions.numpy()
        
    except Exception as e:
        st.error(f"Erreur Grad-CAM: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None, None

# VERSION ALTERNATIVE - Plus robuste
def make_gradcam_heatmap_robuste(img_array, model, layer_name, pred_index=None):
    """
    Version alternative plus robuste pour Grad-CAM.
    """
    try:
        # 1. Obtenir la couche
        layer = model.get_layer(layer_name)
        
        # 2. Créer un modèle pour les sorties de cette couche
        grad_model = keras.models.Model(
            inputs=[model.inputs],
            outputs=[layer.output, model.output]
        )
        
        # 3. Enregistrer les opérations avec GradientTape
        with tf.GradientTape() as tape:
            layer_output, predictions = grad_model(img_array)
            
            # Déterminer l'index de la classe si non fourni
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            
            # CORRECTION: Utiliser tf.gather pour éviter l'indexation par tuple
            class_output = tf.gather(predictions[0], pred_index)
        
        # 4. Calculer les gradients
        grads = tape.gradient(class_output, layer_output)
        
        if grads is None:
            # Fallback: utiliser les activations moyennes
            heatmap = tf.reduce_mean(layer_output[0], axis=-1)
            return heatmap.numpy(), predictions.numpy()
        
        # 5. Pooling des gradients
        grads = tf.reduce_mean(grads, axis=[0, 1, 2])
        
        # 6. Calculer la heatmap
        layer_output = layer_output[0]
        heatmap = tf.reduce_sum(layer_output * grads, axis=-1)
        
        # 7. ReLU et normalisation
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.reduce_max(heatmap)
        
        if max_val > 0:
            heatmap = heatmap / max_val
        
        return heatmap.numpy(), predictions.numpy()
        
    except Exception as e:
        st.error(f"Erreur dans la version robuste: {str(e)}")
        return None, None

# FONCTION POUR SUPERPOSER GRAD-CAM
def superposer_gradcam_simple(img, heatmap, alpha=0.4):
    """
    Superpose la heatmap Grad-CAM sur l'image.
    """
    try:
        # Redimensionner la heatmap pour correspondre à l'image
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        # Normaliser la heatmap entre 0 et 255
        heatmap = np.uint8(255 * heatmap)
        
        # Appliquer la colormap
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Convertir l'image en BGR pour OpenCV
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Superposer
        superimposed = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_colored, alpha, 0)
        
        # Reconvertir en RGB pour l'affichage
        superimposed = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(superimposed)
        
    except Exception as e:
        st.error(f"Erreur de superposition: {str(e)}")
        return Image.fromarray(img)

# FONCTIONS D'AFFICHAGE
def afficher_predictions_detailees(predictions):
    """
    Affiche les probabilités de prédiction.
    """
    st.subheader("📊 Probabilités détaillées")
    
    cols = st.columns(4)
    for i, classe in enumerate(NOMS_CLASSES):
        proba = predictions[i] * 100
        with cols[i]:
            # Déterminer la couleur
            if proba > 70:
                color = "🟢"
            elif proba > 30:
                color = "🟡"
            else:
                color = "⚪"
            
            st.markdown(f"**{classe.capitalize()}** {color}")
            st.progress(float(proba/100), text=f"{proba:.1f}%")

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
    st.sidebar.header("⚙️ Paramètres")
    
    # Charger le modèle
    model = charger_modele()
    
    # Afficher les informations du modèle
    with st.sidebar.expander("📋 Informations du modèle"):
        st.write(f"**Nombre de couches:** {len(model.layers)}")
        st.write("**Dernières couches:**")
        for i, layer in enumerate(model.layers[-5:]):
            st.write(f"- {layer.name} ({layer.__class__.__name__})")
    
    # Paramètre de transparence
    alpha = st.sidebar.slider("Transparence Grad-CAM", 0.1, 0.8, 0.4, 0.1)
    
    # Téléchargement d'image
    st.header("📤 Téléchargement d'image")
    
    uploaded_file = st.file_uploader(
        "Choisissez une image IRM cérébrale",
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        try:
            # Ouvrir l'image
            image = Image.open(uploaded_file).convert("RGB")
            original_image = image.copy()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🖼️ Image originale")
                st.image(image, use_container_width=True)
                st.caption(f"Dimensions: {image.size}")
            
            with col2:
                st.subheader("🔍 Zone d'analyse")
                
                if st.button("🚀 Lancer l'analyse complète", type="primary", use_container_width=True):
                    with st.spinner("Analyse en cours..."):
                        # Préparer l'image
                        img_array = preparer_image(image)
                        
                        # Prédiction
                        predictions = model.predict(img_array, verbose=0)[0]
                        pred_index = np.argmax(predictions)
                        predicted_class = NOMS_CLASSES[pred_index]
                        confidence = predictions[pred_index] * 100
                        
                        # Afficher résultats
                        st.success(f"**Diagnostic prédit: {predicted_class.capitalize()}**")
                        st.info(f"**Confiance: {confidence:.1f}%**")
                        
                        # Essayer différentes méthodes pour Grad-CAM
                        heatmap = None
                        method_used = ""
                        
                        # Méthode 1: Version simple
                        st.write("Tentative avec la méthode simple...")
                        heatmap, _ = make_gradcam_heatmap_simple(
                            img_array, model, NOM_DERNIERE_COUCHE_CONV, pred_index
                        )
                        
                        if heatmap is not None:
                            method_used = "méthode simple"
                        else:
                            # Méthode 2: Version robuste
                            st.write("Tentative avec la méthode robuste...")
                            heatmap, _ = make_gradcam_heatmap_robuste(
                                img_array, model, NOM_DERNIERE_COUCHE_CONV, pred_index
                            )
                            if heatmap is not None:
                                method_used = "méthode robuste"
                            else:
                                # Méthode 3: Essayer d'autres couches
                                st.write("Recherche d'une couche alternative...")
                                for layer in model.layers:
                                    if 'conv' in layer.name.lower() or layer.name != NOM_DERNIERE_COUCHE_CONV:
                                        try:
                                            heatmap, _ = make_gradcam_heatmap_simple(
                                                img_array, model, layer.name, pred_index
                                            )
                                            if heatmap is not None:
                                                method_used = f"couche alternative: {layer.name}"
                                                break
                                        except:
                                            continue
                        
                        if heatmap is not None:
                            # Préparer l'image pour superposition
                            img_np = np.array(original_image.resize(TAILLE_IMAGE))
                            
                            # Superposer Grad-CAM
                            superimposed_img = superposer_gradcam_simple(
                                img_np, heatmap, alpha
                            )
                            
                            # Afficher résultat
                            st.image(
                                superimposed_img,
                                caption=f"Visualisation Grad-CAM ({method_used})",
                                use_container_width=True
                            )
                            
                            # Légende
                            with st.expander("🎨 Légende des couleurs"):
                                st.markdown("""
                                - **🟥 Rouge vif**: Zones les plus importantes pour la décision
                                - **🟨 Jaune/Orange**: Zones moyennement importantes
                                - **🟦 Bleu**: Zones moins importantes
                                
                                *La chaleur de la couleur indique l'importance de la région pour la classification.*
                                """)
                            
                            # Détails des prédictions
                            afficher_predictions_detailees(predictions)
                            
                        else:
                            st.warning("⚠️ Grad-CAM non disponible")
                            st.info("Affichage des prédictions uniquement:")
                            afficher_predictions_detailees(predictions)
                            
                            # Alternative: afficher juste l'image avec les prédictions
                            st.subheader("📈 Prédictions")
                            for i, classe in enumerate(NOMS_CLASSES):
                                proba = predictions[i] * 100
                                st.write(f"{classe.capitalize()}: {proba:.1f}%")
        
        except Exception as e:
            st.error(f"❌ Erreur: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    else:
        # Mode démo
        st.info("👆 Veuillez télécharger une image IRM pour commencer")
        
        with st.expander("ℹ️ Instructions"):
            st.markdown("""
            1. **Téléchargez** une image IRM cérébrale
            2. **Cliquez** sur "Lancer l'analyse complète"
            3. **Analysez** les résultats et visualisez les zones importantes
            
            **Formats acceptés:** JPG, JPEG, PNG
            """)
    
    # Pied de page
    st.divider()
    st.caption("🔬 Application de recherche - À utiliser comme outil d'aide au diagnostic")
    st.caption("⚠️ Les résultats doivent être validés par un professionnel de santé")

if __name__ == "__main__":
    main()
