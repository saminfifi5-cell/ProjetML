# app.py
import streamlit as st
import requests

st.title("Prédiction d'octroi de prêt bancaire")
st.markdown("Remplissez le dossier client pour obtenir une prédiction.")

# Formulaire
col1, col2 = st.columns(2)

with col1:
    age           = st.number_input("Âge", min_value=18, max_value=100, value=35)
    income        = st.number_input("Revenu annuel (€)", min_value=0, value=40000)
    loan_amount   = st.number_input("Montant du prêt (€)", min_value=0, value=20000)
    credit_score  = st.number_input("Score de crédit", min_value=300, max_value=850, value=650)
    months_emp    = st.number_input("Mois d'emploi", min_value=0, value=24)
    credit_lines  = st.number_input("Nombre de lignes de crédit", min_value=0, value=3)
    marital        = st.selectbox("Statut marital", ["Single", "Married", "Divorced"])
    loan_purpose   = st.selectbox("Objet du prêt", ["Auto", "Business", "Education", "Home", "Other"])

with col2:
    interest_rate = st.number_input("Taux d'intérêt (%)", min_value=0.0, value=5.0)
    loan_term     = st.selectbox("Durée du prêt (mois)", [12, 24, 36, 48, 60])
    dti_ratio     = st.number_input("Ratio dette/revenu", min_value=0.0, max_value=1.0, value=0.3)
    has_mortgage  = st.selectbox("Propriétaire ?", ["Non", "Oui"])
    has_dependents= st.selectbox("Personnes à charge ?", ["Non", "Oui"])
    has_cosigner  = st.selectbox("Co-signataire ?", ["Non", "Oui"])
    education      = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
    employment     = st.selectbox("Type d'emploi", ["Full-time", "Part-time", "Self-employed", "Unemployed"])



# Bouton de prédiction
if st.button("Analyser le dossier"):
    payload = {
        "Age": age,
        "Income": income,
        "LoanAmount": loan_amount,
        "CreditScore": credit_score,
        "MonthsEmployed": months_emp,
        "NumCreditLines": credit_lines,
        "InterestRate": interest_rate,
        "LoanTerm": loan_term,
        "DTIRatio": dti_ratio,
        "HasMortgage": 1 if has_mortgage == "Oui" else 0,
        "HasDependents": 1 if has_dependents == "Oui" else 0,
        "HasCoSigner": 1 if has_cosigner == "Oui" else 0,


        # Education (4 modalités)
        "Education_Bachelor's": 1 if education == "Bachelor's" else 0,
        "Education_High School": 1 if education == "High School" else 0,
        "Education_Master's":    1 if education == "Master's" else 0,
        "Education_PhD":         1 if education == "PhD" else 0,

        # EmploymentType (4 modalités)
        "EmploymentType_Full-time":     1 if employment == "Full-time" else 0,
        "EmploymentType_Part-time":     1 if employment == "Part-time" else 0,
        "EmploymentType_Self-employed": 1 if employment == "Self-employed" else 0,
        "EmploymentType_Unemployed":    1 if employment == "Unemployed" else 0,

        # MaritalStatus (3 modalités)
        "MaritalStatus_Divorced": 1 if marital == "Divorced" else 0,
        "MaritalStatus_Married":  1 if marital == "Married" else 0,
        "MaritalStatus_Single":   1 if marital == "Single" else 0,

        # LoanPurpose (5 modalités)
        "LoanPurpose_Auto":      1 if loan_purpose == "Auto" else 0,
        "LoanPurpose_Business":  1 if loan_purpose == "Business" else 0,
        "LoanPurpose_Education": 1 if loan_purpose == "Education" else 0,
        "LoanPurpose_Home":      1 if loan_purpose == "Home" else 0,
        "LoanPurpose_Other":     1 if loan_purpose == "Other" else 0,
    }

    response = requests.post("http://localhost:8000/predict", json=payload)
    print("Status code:", response.status_code)
    print("Réponse brute:", response.text)
    result   = response.json()

    # Affichage du résultat
    st.divider()
    proba_defaut = result['probabilite']

    if proba_defaut >= 0.5:
        st.error(f"⚠️ Risque de **défaut** — Probabilité de défaut : {proba_defaut*100:.1f}%")
    else:
        st.success(f"✅ Profil **fiable** — Probabilité de défaut : {proba_defaut*100:.1f}%")


#model_choice = st.selectbox(
#    "Modèle de prédiction",
#    options=["logistic", "decision_tree"],
#    format_func=lambda x: {
#        "logistic":      "📈 Régression Logistique",
#        "decision_tree": "🌳 Arbre de Décision",
#    }[x]
#)