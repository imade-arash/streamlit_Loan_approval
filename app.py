import streamlit as st
import pandas as pd
import numpy as np
import pickle

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
        raw_features = np.array([[no_of_dependents, income_annum, loan_amount, loan_term, cibil_score,
                                  residential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value,
                                  education_graduate, education_not_graduate, self_employed_no, self_employed_yes]])


        prediction = model.predict(raw_features)[0]

        if prediction == 1:
            st.success("✅ Le prêt est probablement approuvé.")
        else:
            st.info("❌ Le prêt risque d’être refusé.")

with tab2:
    
    # Télécharger les données depuis un fichier CSV
    df = pd.read_csv("C:/Users/HP/Documents/machine learning/SUPERVISED/TD_2/CC3/loan_approval_dataset.csv") 
    df.columns = df.columns.str.strip()

    import matplotlib.pyplot as plt
    import seaborn as sns

    st.write("Quelle est la distribution des cibil_score ?")
    fig, ax = plt.subplots()
    sns.histplot(df['cibil_score'], bins=20, kde=True, ax=ax)
    ax.set_title('Distribution du Score CIBIL')
    st.pyplot(fig)


    st.write(" moyenne des actifs bancaires par statut de prêt ")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x='loan_status', y='bank_asset_value', palette='viridis', ax=ax2)
    ax2.set_title("Valeur moyenne des actifs bancaires par statut de prêt")
    ax2.set_ylabel("Valeur des actifs bancaires")
    ax2.set_xlabel("Statut du prêt")
    st.pyplot(fig2)


    st.write(" Montant du prêt selon l'emploi ")
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x='self_employed', y='loan_amount', palette='coolwarm', ax=ax4)
    ax4.set_title("Montant du prêt selon le statut d'emploi")
    ax4.set_xlabel("Auto-entrepreneur")
    ax4.set_ylabel("Montant du prêt")
    st.pyplot(fig4)


    st.write("  répartition Graduate/Not Graduate ? ")
    fig5, ax5 = plt.subplots()
    df['education'].value_counts().plot.pie(
        autopct='%1.1f%%',
        startangle=90,
        ax=ax5
    )
    ax5.set_title('Répartition Graduate/Not Graduate')
    ax5.set_ylabel('')  
    st.pyplot(fig5)


    st.write("Répartition no_of_dependents des clients")
    fig6, ax6 = plt.subplots(figsize=(6, 4))
    sns.countplot(x='no_of_dependents', data=df, ax=ax6)
    ax6.set_title('Répartition des dépendants des clients')
    ax6.set_xlabel("Nombre de personnes à charge")
    ax6.set_ylabel("Nombre de clients")
    st.pyplot(fig6)


    st.write("Histogramme du commercial_assets_value")
    fig7, ax7 = plt.subplots(figsize=(6, 4))
    sns.histplot(df['commercial_assets_value'], bins=10, kde=True, ax=ax7)
    ax7.set_title('Distribution des valeurs des actifs commerciaux')
    ax7.set_xlabel("Valeur des actifs commerciaux")
    ax7.set_ylabel("Fréquence")
    st.pyplot(fig7)


    st.write(" Répartition loan_status des clients")
    fig8, ax8 = plt.subplots(figsize=(6, 4))
    sns.countplot(x='education', hue='loan_status', data=df, ax=ax8)
    ax8.set_title('Répartition du statut de prêt selon le niveau d\'éducation')
    ax8.set_xlabel("Niveau d'éducation")
    ax8.set_ylabel("Nombre de clients")
    st.pyplot(fig8)


    st.write("Distribution des scores CIBIL selon le statut du prêt")
    fig9, ax9 = plt.subplots(figsize=(6, 4))
    sns.histplot(data=df, x='cibil_score', hue='loan_status', kde=True, palette='Set1', ax=ax9)
    ax9.set_title("Distribution des scores CIBIL selon le statut du prêt")
    ax9.set_xlabel("Score CIBIL")
    ax9.set_ylabel("Fréquence")
    st.pyplot(fig9)


    st.write("Répartition des termes du prêt selon le statut du prêt")
    fig5, ax5 = plt.subplots(figsize=(6, 4))
    sns.countplot(x='loan_term', hue='loan_status', data=df, ax=ax5)
    ax5.set_title("Répartition des termes du prêt selon le statut du prêt")
    ax5.set_xlabel("Durée du prêt (années)")
    ax5.set_ylabel("Nombre de clients")
    st.pyplot(fig5)


    st.write("📊 Pourcentage des statuts de prêt")
    loan_status_counts = df['loan_status'].value_counts()
    fig11, ax11 = plt.subplots()
    ax11.pie(loan_status_counts, labels=loan_status_counts.index, autopct='%1.1f%%', startangle=90, colors=['#8fd9b6', '#ff9999'])
    ax11.set_title("Répartition des statuts de prêt")
    st.pyplot(fig11)


    st.write("📊 Nombre de clients selon la durée du prêt")
    fig15, ax15 = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x='loan_term', palette='Set3', ax=ax15)
    ax15.set_title("Nombre de clients par durée de prêt")
    ax15.set_xlabel("Durée du prêt (en mois)")
    ax15.set_ylabel("Nombre de clients")
    st.pyplot(fig15)


    st.write("Répartition des clients par catégorie de score CIBIL")
    df['cibil_cat'] = pd.cut(df['cibil_score'], bins=[300, 500, 650, 750, 900],
                         labels=['Faible', 'Moyen', 'Bon', 'Excellent'])

    fig17, ax17 = plt.subplots()
    sns.countplot(x='cibil_cat', data=df, palette='YlOrBr', ax=ax17)
    ax17.set_title("Répartition des clients par catégorie de score CIBIL")
    ax17.set_xlabel("Catégorie de score CIBIL")
    ax17.set_ylabel("Nombre de clients")
    st.pyplot(fig17)


    st.write("Tendance entre la durée du prêt et son montant")
    fig20, ax20 = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=df.sort_values(by='loan_term'), x='loan_term', y='loan_amount', marker='o', ax=ax20)
    ax20.set_title("Tendance entre la durée du prêt et son montant")
    ax20.set_xlabel("Durée du prêt")
    ax20.set_ylabel("Montant du prêt")
    st.pyplot(fig20)
