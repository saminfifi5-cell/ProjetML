# api.py
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle
# from decisionTree.essaie_implementation_interface.ArbreDeDecision import DecisionTree

app = FastAPI()
#models = {} 

# Chargement du modèle au démarrage
with open("models/logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

#with open("models/arbre_fait_maison.pkl", "rb") as f:
#    models["decision_tree"] = pickle.load(f)

w    = model['weights']
mean = model['mean']
std  = model['std']

# Schéma des données entrantes
class DossierClient(BaseModel):
    # model_type: str = "logistic"
    Age: int
    Income: float
    LoanAmount: float
    CreditScore: int
    MonthsEmployed: int
    NumCreditLines: int
    InterestRate: float
    LoanTerm: int
    DTIRatio: float
    HasMortgage: int
    HasDependents: int
    HasCoSigner: int
    Education_Bachelors: int = 0
    Education_High_School: int = 0
    Education_Masters: int = 0
    Education_PhD: int = 0
    EmploymentType_Full_time: int = 0
    EmploymentType_Part_time: int = 0
    EmploymentType_Self_employed: int = 0
    EmploymentType_Unemployed: int = 0
    MaritalStatus_Divorced: int = 0
    MaritalStatus_Married: int = 0
    MaritalStatus_Single: int = 0
    LoanPurpose_Auto: int = 0
    LoanPurpose_Business: int = 0
    LoanPurpose_Education: int = 0
    LoanPurpose_Home: int = 0
    LoanPurpose_Other: int = 0

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


#@app.post("/predict")
#def predict(dossier: DossierClient):
    model_type = dossier.model_type

    if model_type not in models:
        return {"error": f"Modèle '{model_type}' non disponible"}

    data = dossier.dict()
    data.pop("model_type")
    x = np.array([list(data.values())], dtype=float)

    m = models[model_type]
    x = (x - m["mean"]) / m["std"]

    if model_type == "logistic":
        x_bias = np.hstack([np.ones((1, 1)), x])
        proba = sigmoid(x_bias @ m["weights"])[0]

    elif model_type == "decision_tree":
        proba = m["model"].predict_proba(x)[0][1]

    return {
        "probabilite": round(float(proba), 3),
        "decision": "Défaut probable" if proba >= 0.5 else "Pas de défaut",
        "modele_utilise": model_type
    }


@app.post("/predict")
def predict(dossier: DossierClient):
    # Construire le vecteur de features
    x = np.array([list(dossier.dict().values())], dtype=float)

    # Normalisation avec les params du train
    x = (x - mean) / std

    # Ajout du biais
    x = np.hstack([np.ones((1, 1)), x])

    # Prédiction
    proba = sigmoid(x @ w)[0]
    decision = "Accordé" if proba >= 0.5 else "Refusé"

    return {
        "probabilite": round(float(proba), 3),
        "decision": "Défaut probable" if proba >= 0.5 else "Pas de défaut"
    }

# Lancer avec : uvicorn api:app --reload