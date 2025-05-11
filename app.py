import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Charger le modèle
model = pickle.load(open("C:/Users/HP/Documents/machine learning/SUPERVISED/TD_2/CC3/DT_Loan_approval.pkl", "rb"))

st.title("🔍 Prédiction d'Approbation de Prêt")

tab1, tab2 = st.tabs(["📊 Prédiction de l’approbation du prêt", "📈 Analyse des données"])

with tab1:
    st.write("Veuillez remplir les informations ci-dessous pour prédire si le prêt sera approuvé :")

    # --- مدخلات المستخدم ---
    no_of_dependents = st.number_input("Nombre de personnes à charge", min_value=0, max_value=10, value=1)
    income_annum = st.number_input("Revenu annuel (en unités monétaires)", value=5000000.0)
    loan_amount = st.number_input("Montant du prêt demandé", value=10000000.0)
    loan_term = st.number_input("Durée du prêt (en années)", value=10)
    cibil_score = st.slider("Score CIBIL", min_value=300, max_value=900, value=600)

    residential_assets_value = st.number_input("Valeur des biens résidentiels", value=2000000.0)
    commercial_assets_value = st.number_input("Valeur des biens commerciaux", value=3000000.0)
    luxury_assets_value = st.number_input("Valeur des biens de luxe", value=5000000.0)
    bank_asset_value = st.number_input("Valeur des actifs bancaires", value=1000000.0)

    # --- Encodage de l'éducation ---
    education = st.selectbox("Niveau d'éducation", options=["Graduate", "Not Graduate"])
    education_graduate = 1 if education == "Graduate" else 0
    education_not_graduate = 1 if education == "Not Graduate" else 0

    # --- Encodage emploi ---
    self_employed = st.selectbox("Travailleur indépendant ?", options=["Oui", "Non"])
    self_employed_yes = 1 if self_employed == "Oui" else 0
    self_employed_no = 1 if self_employed == "Non" else 0

    # --- Bouton de prédiction ---
    if st.button("📊 Prédire"):
        features = np.array([[no_of_dependents, income_annum, loan_amount, loan_term, cibil_score,
                              residential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value,
                              education_graduate, education_not_graduate, self_employed_no, self_employed_yes]])

        prediction = model.predict(features)[0]

        if prediction == 1:
            st.success("✅ Le prêt est probablement approuvé.")
        else:
            st.info("❌ Le prêt risque d’être refusé.")

# with tab2:
#     st.subheader("📄 Rapport de l’analyse précédente")
#     try:
#         with open("CC3_html.html", "r", encoding="utf-8") as f:
#             html_data = f.read()
#         st.components.v1.html(html_data, height=800, scrolling=True)
#     except FileNotFoundError:
#         st.error("❗ Le fichier d’analyse CC3_html.html est introuvable dans le dossier actuel.")