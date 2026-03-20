import streamlit as st

import numpy as np

import tensorflow as tf

from tensorflow import keras

from PIL import Image

import os

import hashlib

import hmac



# CONFIGURATION DE L'AUTHENTIFICATION

# Les mots de passe hachés peuvent être définis via les variables d'environnement
# APP_ADMIN_HASH et APP_MEDECIN_HASH. Par défaut, des valeurs de démonstration sont utilisées.
# Pour générer un hachage : hashlib.sha256(b"votre_mot_de_passe").hexdigest()

_HASH_ADMIN_PAR_DEFAUT = "240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a"
_HASH_MEDECIN_PAR_DEFAUT = "e7d2f9e5af4a1b2c3d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7"

UTILISATEURS = {
    "admin": os.environ.get("APP_ADMIN_HASH", _HASH_ADMIN_PAR_DEFAUT),
    "medecin": os.environ.get("APP_MEDECIN_HASH", _HASH_MEDECIN_PAR_DEFAUT),
}



def verifier_mot_de_passe(nom_utilisateur, mot_de_passe):

    """

    Vérifie le nom d'utilisateur et le mot de passe en utilisant une comparaison
    à temps constant pour prévenir les attaques temporelles.

    """

    hash_attendu = UTILISATEURS.get(nom_utilisateur)

    if hash_attendu is None:

        # Effectuer un hachage bidon pour éviter l'énumération de comptes par timing
        hashlib.sha256(mot_de_passe.encode()).hexdigest()

        return False

    hash_fourni = hashlib.sha256(mot_de_passe.encode()).hexdigest()

    return hmac.compare_digest(hash_attendu, hash_fourni)



def afficher_page_connexion():

    """

    Affiche la page de connexion.

    """

    st.markdown("""

    <style>

    .login-container {

        max-width: 400px;

        margin: auto;

        padding: 40px;

        border-radius: 10px;

        background-color: #f8f9fa;

        border: 1px solid #dee2e6;

    }

    </style>

    """, unsafe_allow_html=True)

    st.markdown('<h1 style="text-align: center; color: #2c3e50;">🧠 Système d\'Analyse d\'IRM</h1>', unsafe_allow_html=True)

    st.markdown('<h3 style="text-align: center; color: #7f8c8d; margin-bottom: 30px;">Connexion</h3>', unsafe_allow_html=True)

    with st.form("formulaire_connexion"):

        nom_utilisateur = st.text_input("Nom d'utilisateur", placeholder="Entrez votre nom d'utilisateur")

        mot_de_passe = st.text_input("Mot de passe", type="password", placeholder="Entrez votre mot de passe")

        soumettre = st.form_submit_button("Se connecter", use_container_width=True, type="primary")

        if soumettre:

            if verifier_mot_de_passe(nom_utilisateur, mot_de_passe):

                st.session_state["connecte"] = True

                st.session_state["utilisateur"] = nom_utilisateur

                st.rerun()

            else:

                st.error("Identifiants invalides. Veuillez réessayer.")



# CONFIGURATION DES CONSTANTES

TAILLE_IMAGE = (224, 224)

NOMS_CLASSES = ['Glioma', 'Méningiome', 'Pas de tumeur', 'Hypophyse']

DESCRIPTIONS_CLASSES = {

    'Glioma': 'Tumeur se développant dans le tissu glial du cerveau',

    'Méningiome': 'Tumeur se développant dans les méninges',

    'Pas de tumeur': 'Aucune tumeur détectée sur l\'image',

    'Hypophyse': 'Tumeur de la glande hypophysaire'

}

CHEMIN_MODELE = "modele.h5"



# FONCTIONS DE CHARGEMENT ET PRÉTRAITEMENT



@st.cache_resource

def charger_modele():

    """

    Charge le modèle Keras pré-entraîné.

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

            st.error("Fichier modèle introuvable")

            st.info("Veuillez vérifier que le fichier modèle est présent dans le dépôt")

            st.stop()

        

        modele = keras.models.load_model(chemin_trouve, compile=False)

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



def afficher_resultats(predictions):

    """

    Affiche les résultats de prédiction de manière professionnelle.

    """

    # Trouver la prédiction principale

    idx_principal = np.argmax(predictions)

    classe_principale = NOMS_CLASSES[idx_principal]

    confiance_principale = predictions[idx_principal] * 100

    

    # Créer deux colonnes pour l'affichage

    col1, col2 = st.columns([1, 2])

    

    with col1:

        # Afficher le diagnostic principal

        st.markdown("### Diagnostic")

        

        # Déterminer la couleur en fonction de la confiance

        if confiance_principale >= 90:

            couleur = "#28a745"  # Vert

            emoji = "✅"

        elif confiance_principale >= 70:

            couleur = "#ffc107"  # Jaune

            emoji = "⚠️"

        else:

            couleur = "#dc3545"  # Rouge

            emoji = "❓"

        

        st.markdown(f"""

        <div style="border-left: 4px solid {couleur}; padding-left: 15px; margin: 10px 0;">

            <h4 style="color: {couleur}; margin-bottom: 5px;">{classe_principale} {emoji}</h4>

            <p style="font-size: 24px; font-weight: bold; color: {couleur};">

                {confiance_principale:.1f}%

            </p>

        </div>

        """, unsafe_allow_html=True)

    

    with col2:

        # Afficher la description de la classe

        st.markdown("### Description")

        st.info(DESCRIPTIONS_CLASSES[classe_principale])

    

    # Séparateur

    st.divider()

    

    # Afficher toutes les probabilités

    st.markdown("### Probabilités détaillées")

    

    # Créer un tableau pour les probabilités

    cols = st.columns(4)

    

    for i, classe in enumerate(NOMS_CLASSES):

        proba = predictions[i] * 100

        

        with cols[i]:

            # Créer une barre de progression personnalisée

            progress_html = f"""

            <div style="margin-bottom: 15px;">

                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">

                    <span style="font-weight: bold;">{classe}</span>

                    <span style="color: #6c757d;">{proba:.1f}%</span>

                </div>

                <div style="background: #e9ecef; height: 8px; border-radius: 4px; overflow: hidden;">

                    <div style="background: {'#28a745' if i == idx_principal else '#007bff'}; 

                         width: {proba}%; height: 100%;">

                    </div>

                </div>

            </div>

            """

            st.markdown(progress_html, unsafe_allow_html=True)



def afficher_guide_medical():

    """

    Affiche le guide médical dans la sidebar.

    """

    with st.sidebar.expander("Guide médical"):

        st.markdown("""

        **Classes diagnostiques :**

        

        **Glioma**

        - Tumeur du tissu glial cérébral

        - Peut être bénigne ou maligne

        - Localisation variable dans le cerveau

        

        **Méningiome**

        - Tumeur des méninges

        - Généralement bénigne

        - Croissance lente

        

        **Pas de tumeur**

        - Absence de tumeur détectée

        - Image normale ou pathologie non tumorale

        

        **Hypophyse**

        - Tumeur de la glande hypophysaire

        - Peut affecter la production hormonale

        - Localisation : selle turcique

        

        **Note importante :**

        Cette application fournit une analyse préliminaire.

        Tout résultat doit être validé par un radiologue qualifié.

        """)



# APPLICATION PRINCIPALE



def main():

    """

    Fonction principale de l'application.

    """

    # Configuration de la page (doit être appelée une seule fois, en premier)

    st.set_page_config(

        page_title="Système d'Analyse d'IRM Cérébrales",

        page_icon="🧠",

        layout="wide",

        initial_sidebar_state="expanded"

    )

    

    # Vérification de l'authentification

    if "connecte" not in st.session_state or not st.session_state["connecte"]:

        afficher_page_connexion()

        return

    

    # CSS personnalisé

    st.markdown("""

    <style>

    .main-header {

        color: #2c3e50;

        border-bottom: 2px solid #3498db;

        padding-bottom: 10px;

        margin-bottom: 30px;

    }

    .diagnostic-card {

        background-color: #f8f9fa;

        border-radius: 10px;

        padding: 20px;

        border-left: 5px solid #3498db;

        margin: 20px 0;

    }

    .upload-section {

        background-color: #e8f4f8;

        padding: 20px;

        border-radius: 10px;

        margin: 20px 0;

    }

    .footer {

        text-align: center;

        color: #7f8c8d;

        font-size: 12px;

        margin-top: 50px;

        padding-top: 20px;

        border-top: 1px solid #ecf0f1;

    }

    </style>

    """, unsafe_allow_html=True)

    

    # En-tête

    st.markdown('<h1 class="main-header">Système d\'Analyse d\'IRM Cérébrales</h1>', unsafe_allow_html=True)

    st.markdown("""

    <div style="color: #34495e; font-size: 16px; margin-bottom: 30px;">

    Application d'aide au diagnostic basée sur l'intelligence artificielle.

    Classification automatique des images IRM cérébrales en quatre catégories.

    </div>

    """, unsafe_allow_html=True)

    

    # Sidebar

    st.sidebar.markdown("<h3 style='color: #2c3e50;'>Configuration</h3>", unsafe_allow_html=True)

    st.sidebar.markdown(f"👤 Connecté en tant que **{st.session_state.get('utilisateur', '')}**")

    if st.sidebar.button("Se déconnecter", use_container_width=True):

        st.session_state.pop("connecte", None)

        st.session_state.pop("utilisateur", None)

        st.rerun()

    

    # Charger le modèle

    with st.sidebar:

        with st.spinner("Chargement du modèle..."):

            model = charger_modele()

        st.success("Modèle chargé avec succès")

    

    # Informations techniques

    with st.sidebar.expander("Informations techniques"):

        st.write(f"**Résolution d'entrée :** {TAILLE_IMAGE[0]}x{TAILLE_IMAGE[1]} pixels")

        st.write(f"**Architecture :** Réseau de neurones convolutif")

        st.write(f"**Classes :** 4 catégories diagnostiques")

    

    # Guide médical

    afficher_guide_medical()

    

    # Section principale

    st.markdown('<div class="upload-section">', unsafe_allow_html=True)

    st.markdown("### Téléchargement d'image")

    

    uploaded_file = st.file_uploader(

        "Sélectionnez une image IRM cérébrale",

        type=["jpg", "jpeg", "png"],

        help="Formats supportés : JPG, JPEG, PNG"

    )

    st.markdown('</div>', unsafe_allow_html=True)

    

    if uploaded_file is not None:

        try:

            # Ouvrir l'image

            image = Image.open(uploaded_file).convert("RGB")

            

            # Créer deux colonnes pour l'affichage

            col1, col2 = st.columns([1, 1])

            

            with col1:

                st.markdown("### Image originale")

                st.image(image, use_container_width=True)

                st.caption(f"Dimensions : {image.size[0]} x {image.size[1]} pixels")

            

            with col2:

                st.markdown("### Analyse")

                

                if st.button("Lancer l'analyse", type="primary", use_container_width=True):

                    with st.spinner("Analyse en cours..."):

                        # Préparer l'image

                        img_array = preparer_image(image)

                        

                        # Prédiction

                        predictions = model.predict(img_array, verbose=0)[0]

                        

                        # Afficher les résultats

                        afficher_resultats(predictions)

                        

                        # Recommandations médicales

                        st.markdown('<div class="diagnostic-card">', unsafe_allow_html=True)

                        st.markdown("### Recommandations")

                        

                        idx_principal = np.argmax(predictions)

                        classe_principale = NOMS_CLASSES[idx_principal]

                        

                        if classe_principale == "Pas de tumeur":

                            st.success("Aucune action immédiate requise. Suivi recommandé selon protocole standard.")

                        else:

                            st.warning("""

                            **Actions recommandées :**

                            1. Consultation avec un neuro-radiologue

                            2. Examens complémentaires si nécessaire

                            3. Discussion en réunion de concertation pluridisciplinaire

                            4. Planification du suivi

                            """)

                        

                        st.markdown('</div>', unsafe_allow_html=True)

            

            # Informations supplémentaires

            with st.expander("Informations sur l'image"):

                st.write(f"**Format :** {image.format if image.format else 'Inconnu'}")

                st.write(f"**Mode :** {image.mode}")

                st.write("**Note :** L'image a été redimensionnée à 224x224 pixels pour l'analyse")

        

        except Exception as e:

            st.error(f"Erreur lors du traitement de l'image : {str(e)}")

            

            # Afficher des informations de débogage

            with st.expander("Détails de l'erreur"):

                import traceback

                st.code(traceback.format_exc())

    

    else:

        # Message d'accueil

        st.markdown("""

        <div style="background-color: #f0f8ff; padding: 30px; border-radius: 10px; text-align: center; margin-top: 30px;">

            <h3 style="color: #2c3e50;">Bienvenue</h3>

            <p style="color: #34495e; font-size: 16px;">

                Téléchargez une image IRM cérébrale pour obtenir une analyse automatique.

            </p>

            <p style="color: #7f8c8d; font-size: 14px;">

                L'application classifie les images en quatre catégories diagnostiques.

            </p>

        </div>

        """, unsafe_allow_html=True)

        

        # Section d'exemple

        with st.expander("Exemples de cas cliniques"):

            col1, col2, col3, col4 = st.columns(4)

            

            with col1:

                st.markdown("**Glioma**")

                st.image("https://via.placeholder.com/150/FF6B6B/FFFFFF?text=Glioma", 

                        caption="Exemple de glioma", width=150)

            

            with col2:

                st.markdown("**Méningiome**")

                st.image("https://via.placeholder.com/150/4ECDC4/FFFFFF?text=Méningiome", 

                        caption="Exemple de méningiome", width=150)

            

            with col3:

                st.markdown("**Sain**")

                st.image("https://via.placeholder.com/150/45B7D1/FFFFFF?text=Sain", 

                        caption="IRM normale", width=150)

            

            with col4:

                st.markdown("**Hypophyse**")

                st.image("https://via.placeholder.com/150/96CEB4/FFFFFF?text=Hypophyse", 

                        caption="Tumeur hypophysaire", width=150)

    

    # Pied de page

    st.markdown("""

    <div class="footer">

        <p>Application développée pour la recherche médicale</p>

        <p>© 2025 - Système d'Aide au Diagnostic</p>

        <p style="font-size: 11px;">

            Cet outil est destiné aux professionnels de santé et ne remplace pas un diagnostic médical complet.

        </p>

    </div>

    """, unsafe_allow_html=True)



# POINT D'ENTRÉE



if __name__ == "__main__":

    main()
