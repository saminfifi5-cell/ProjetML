import subprocess
import time
import sys
import os

DOSSIER_RACINE = "RN_sousdoss"
DOSSIER_TF = os.path.join(DOSSIER_RACINE, 'data_tf_pipelines')
DOSSIER_MODELES = os.path.join(DOSSIER_RACINE, 'models')
DOSSIER_LOGS = os.path.join(DOSSIER_RACINE, 'logs')
DOSSIER_CLEANED = os.path.join(DOSSIER_RACINE, 'data_cleaned')
DOSSIER_TRANSFORMED = os.path.join(DOSSIER_RACINE, 'data_transformed')

# Configuration liant chaque notebook à son fichier/dossier résultant
pipeline_config = [
    {
        "notebook": "1_Cleaner.ipynb", 
        "marqueur": os.path.join(DOSSIER_RACINE, "valeurs_imputation.json")
    },
    {
        "notebook": "2_Transformer.ipynb", 
        "marqueur": os.path.join(DOSSIER_RACINE, "data_transformed", "encodeur.joblib")
    },
    {
        "notebook": "3_Pipeline.ipynb", 
        "marqueur": os.path.join(DOSSIER_RACINE, "data_tf_pipelines", "train")
    },
    {
        "notebook": "4_Modelisation.ipynb", 
        "marqueur": os.path.join(DOSSIER_RACINE, "models", "baseline_mlp.keras")
    },
    {
        "notebook": "5_Evaluation.ipynb", 
        "marqueur": os.path.join(DOSSIER_RACINE, "performance_finale.json")
    }
]

def executer_pipeline_conditionnel():
    print("Démarrage de l'orchestration conditionnelle du pipeline.")
    print("-" * 60)
    
    for etape in pipeline_config:
        notebook = etape["notebook"]
        marqueur = etape["marqueur"]
        
        # Vérification du marqueur pour éviter la ré-exécution
        if os.path.exists(marqueur):
            print(f"Ignoré : {notebook} (L'artefact '{marqueur}' existe déjà)")
            continue
            
        print(f"Exécution du processus : {notebook}")
        debut_processus = time.time()
        
        resultat = subprocess.run([
            sys.executable, "-m", "jupyter", "nbconvert", 
            "--to", "notebook", 
            "--execute", 
            "--inplace", 
            notebook
        ], capture_output=True, text=True)
        
        temps_ecoule = time.time() - debut_processus
        
        if resultat.returncode == 0:
            print(f"Statut : Succès | Temps de traitement : {temps_ecoule:.2f} secondes")
            print("-" * 60)
        else:
            print(f"Statut : Échec critique sur le fichier {notebook}")
            print("Traceback (1500 derniers caractères) :")
            print(resultat.stderr[-1500:])
            print("\nArrêt préventif de l'orchestrateur.")
            sys.exit(1)
            
    print("\nOrchestration terminée.")

if __name__ == "__main__":
    executer_pipeline_conditionnel()