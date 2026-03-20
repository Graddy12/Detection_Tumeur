# Détection de Tumeurs Cérébrales 🧠

Application d'aide au diagnostic par intelligence artificielle pour la classification d'images IRM cérébrales.

## Description

Cette application Streamlit utilise un réseau de neurones convolutif pré-entraîné pour analyser des images IRM cérébrales et les classifier en quatre catégories diagnostiques :

| Classe | Description |
|--------|-------------|
| **Glioma** | Tumeur se développant dans le tissu glial du cerveau |
| **Méningiome** | Tumeur se développant dans les méninges |
| **Hypophyse** | Tumeur de la glande hypophysaire |
| **Pas de tumeur** | Aucune tumeur détectée sur l'image |

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
streamlit run app.py
```

Puis ouvrez votre navigateur à l'adresse indiquée (par défaut `http://localhost:8501`).

1. Chargez une image IRM cérébrale (formats JPG, JPEG ou PNG)
2. Cliquez sur **Lancer l'analyse**
3. Consultez le diagnostic et les probabilités détaillées

## Structure du projet

```
Detection_Tumeur/
├── app.py            # Application Streamlit principale
├── modele/
│   └── modele.h5     # Modèle Keras pré-entraîné
├── requirements.txt  # Dépendances Python
└── README.md
```

## Avertissement

Cet outil est destiné à un usage de recherche et d'aide au diagnostic. Il ne remplace pas l'avis d'un radiologue ou d'un médecin qualifié. Tout résultat doit être validé par un professionnel de santé.
