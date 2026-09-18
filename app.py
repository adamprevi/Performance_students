
# ==============================
# IMPORTATION
# ==============================

import streamlit as st
import joblib as jb
import numpy as np
import pandas as pd


# ==============================
# CONFIGURATION DE LA PAGE
# ==============================

st.set_page_config(
    page_title="Prédiction de la performance des étudiants",
    page_icon="🎓",
    layout="centered"
)


# ==============================
# CHARGEMENT DES ENCODEURS
# ==============================

encoders = jb.load("encoders_reg1.joblib")

# ==============================
# CHARGEMENT DU SCALER
# ==============================

scaler = jb.load("scaler_reg1.joblib")

# ==============================
# CHARGEMENT DU MODÈLE RANDOM FOREST
# ==============================

rf_model = jb.load("rf_model_reg1.joblib")


# ==============================
# TITRE DE L'APPLICATION
# ==============================

st.title("🎓 Prédiction de la performance des étudiants")

st.write(
    """
    Cette application permet de prédire l'indice de performance
    d'un étudiant à partir de ses heures d'étude, ses scores précédents,
    ses activités extrascolaires, ses heures de sommeil et
    le nombre de sujets d'examen pratiqués.
    """
)


# ==============================
# ONGLETS
# ==============================

tab1, tab2 = st.tabs([
    "📊 Prédiction simple",
    "📁 Prédiction multiple"
])


# ==========================================================
# INTERFACE 1 : PRÉDICTION SIMPLE
# ==========================================================

with tab1:

    st.subheader("Prédiction individuelle")

    Hours_Studied = st.number_input(
        "Hours Studied",
        min_value=0.0,
        value=5.0
    )

    Previous_Scores = st.number_input(
        "Previous Scores",
        min_value=0.0,
        value=70.0
    )

    Extracurricular_Activities = st.selectbox(
        "Extracurricular Activities",
        ["Yes", "No"]
    )

    Sleep_Hours = st.number_input(
        "Sleep Hours",
        min_value=0.0,
        value=7.0
    )

    Sample_Question_Papers_Practiced = st.number_input(
        "Sample Question Papers Practiced",
        min_value=0.0,
        value=5.0
    )


    # ==============================
    # BOUTON DE PRÉDICTION
    # ==============================

    if st.button(
        "🔮 Prédire la performance",
        key="prediction_simple"
    ):

        # Encoder la variable catégorielle
        Extracurricular_Activities = encoders[0].transform(
            [Extracurricular_Activities]
        )[0]

        # Créer le vecteur des variables
        x_new = np.array([
            Hours_Studied,
            Previous_Scores,
            Extracurricular_Activities,
            Sleep_Hours,
            Sample_Question_Papers_Practiced
        ])

        # Transformer en tableau 2D
        x_new = x_new.reshape(1, -1)

        # Appliquer le scaler
        x_new = scaler.transform(x_new)

        # Faire la prédiction
        y_pred = rf_model.predict(x_new)

        # Afficher le résultat
        st.success(
            f"Performance Index prédit : {float(y_pred[0]):.2f}"
        )


# ==========================================================
# INTERFACE 2 : PRÉDICTION MULTIPLE
# ==========================================================

with tab2:

    st.subheader("Prédiction à partir d'un fichier CSV")

    st.write(
        """
        Importez un fichier CSV contenant les colonnes suivantes :
        """
    )

    st.code(
        """
Hours Studied
Previous Scores
Extracurricular Activities
Sleep Hours
Sample Question Papers Practiced
        """
    )


    uploaded_file = st.file_uploader(
        "Importer un fichier CSV",
        type=["csv"]
    )


    if uploaded_file is not None:

        # Lire le fichier CSV
        df = pd.read_csv(uploaded_file)

        st.write("### Aperçu des données")
        st.dataframe(df.head())


        # ==============================
        # VÉRIFICATION DES COLONNES
        # ==============================

        required_columns = [
            "Hours Studied",
            "Previous Scores",
            "Extracurricular Activities",
            "Sleep Hours",
            "Sample Question Papers Practiced"
        ]

        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]


        if missing_columns:

            st.error(
                "Colonnes manquantes : "
                + ", ".join(missing_columns)
            )

        else:

            if st.button(
                "🔮 Prédire toutes les performances",
                key="prediction_multiple"
            ):

                predictions = []

                # Parcourir les lignes
                for _, row in df.iterrows():

                    # Encoder la variable catégorielle
                    activity = encoders[0].transform(
                        [row["Extracurricular Activities"]]
                    )[0]

                    # Créer le vecteur
                    x_new = np.array([
                        row["Hours Studied"],
                        row["Previous Scores"],
                        activity,
                        row["Sleep Hours"],
                        row["Sample Question Papers Practiced"]
                    ])

                    # Transformer en tableau 2D
                    x_new = x_new.reshape(1, -1)

                    # Appliquer le scaler
                    x_new = scaler.transform(x_new)

                    # Prédiction
                    y_pred = rf_model.predict(x_new)

                    predictions.append(
                        round(float(y_pred[0]), 2)
                    )


                # Ajouter les prédictions
                df["Performance Index"] = predictions


                # ==============================
                # AFFICHAGE DES RÉSULTATS
                # ==============================

                st.success(
                    "Les prédictions ont été effectuées avec succès."
                )

                st.write("### Résultats")
                st.dataframe(df)


                # ==============================
                # CRÉATION DU FICHIER
                # ==============================

                csv = df.to_csv(index=False).encode("utf-8")


                # ==============================
                # BOUTON DE TÉLÉCHARGEMENT
                # ==============================

                st.download_button(
                    label="⬇️ Télécharger les prédictions",
                    data=csv,
                    file_name="predictions_reg1.csv",
                    mime="text/csv"
                )



