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

with col2:
    interest_rate = st.number_input("Taux d'intérêt (%)", min_value=0.0, value=5.0)
    loan_term     = st.selectbox("Durée du prêt (mois)", [12, 24, 36, 48, 60])
    dti_ratio     = st.number_input("Ratio dette/revenu", min_value=0.0, max_value=1.0, value=0.3)
    has_mortgage  = st.selectbox("Propriétaire ?", ["Non", "Oui"])
    has_dependents= st.selectbox("Personnes à charge ?", ["Non", "Oui"])
    has_cosigner  = st.selectbox("Co-signataire ?", ["Non", "Oui"])

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
    }

    response = requests.post("http://localhost:8000/predict", json=payload)
    result   = response.json()

    # Affichage du résultat
    st.divider()
    if result['decision'] == "Accordé":
        st.success(f"✅ Prêt **Accordé** — Probabilité : {result['probabilite']*100:.1f}%")
    else:
        st.error(f"❌ Prêt **Refusé** — Probabilité : {result['probabilite']*100:.1f}%")
