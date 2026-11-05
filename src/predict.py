"""
Projet 2 — Prédiction avec le modèle entraîné
Auteur : Mame Awa NDIAYE

Utilisation :
    import pandas as pd
    from src.predict import predict
    predictions = predict(pd.read_csv("data/flux_train.csv").drop(columns=["attaque"]))

ou en ligne de commande :  python src/predict.py data/flux_train.csv
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

FICHIER_MODELE = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
_modele = None   # le modèle est chargé une seule fois, au premier appel


def _charger_modele():
    global _modele
    if _modele is None:
        _modele = joblib.load(FICHIER_MODELE)
    return _modele


def predict(df: pd.DataFrame) -> list:
    """
    df : mêmes colonnes que flux_train.csv, SANS 'attaque'.
    Retourne une liste de 0 (normal) ou 1 (attaque), une valeur par ligne.
    """
    modele = _charger_modele()
    X = df.drop(columns=["attaque"], errors="ignore")    # au cas où la colonne cible serait présente
    return [int(p) for p in modele.predict(X)]


if __name__ == "__main__":
    chemin = sys.argv[1] if len(sys.argv) > 1 else "data/flux_train.csv"
    donnees = pd.read_csv(chemin)
    resultat = predict(donnees)
    print(f"{len(resultat)} flux analysés : {sum(resultat)} attaques détectées, "
          f"{len(resultat) - sum(resultat)} flux normaux.")
