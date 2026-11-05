"""
Projet 2 — Détecter des attaques réseau avec le Machine Learning
Auteur : Mame Awa NDIAYE

Lancement (depuis la racine du dépôt) :  python src/train.py

Étapes :
  1. explorer les données (taille, valeurs manquantes, répartition de la cible) ;
  2. construire le prétraitement dans un Pipeline (imputation, mise à l'échelle, encodage) ;
  3. entraîner et comparer 3 modèles sur un jeu de validation ;
  4. choisir le meilleur (F1 de la classe attaque), le ré-entraîner sur toutes les données ;
  5. l'enregistrer dans models/model.joblib.
"""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RACINE = Path(__file__).resolve().parent.parent
FICHIER_DONNEES = RACINE / "data" / "flux_train.csv"
FICHIER_MODELE = RACINE / "models" / "model.joblib"
DOSSIER_FIGURES = RACINE / "figures"

CIBLE = "attaque"
COLONNES_TEXTE = ["protocol"]
COLONNES_NUMERIQUES = ["duration", "dst_port", "src_bytes", "dst_bytes", "nb_packets",
                       "nb_failed_logins", "nb_conn_same_host", "syn_rate"]
RANDOM_STATE = 42


# ----------------------------------------------------------------------------
# 1. Exploration
# ----------------------------------------------------------------------------
def explorer(df: pd.DataFrame) -> None:
    print("=== 1. Exploration ===")
    print(f"Taille : {df.shape[0]} flux, {df.shape[1]} colonnes")
    manquants = df.isna().sum()
    print("Valeurs manquantes :", manquants[manquants > 0].to_dict())
    repartition = df[CIBLE].value_counts(normalize=True).mul(100).round(1)
    print(f"Répartition de '{CIBLE}' : normal = {repartition[0]} %, attaque = {repartition[1]} %")


# ----------------------------------------------------------------------------
# 2. Prétraitement
# ----------------------------------------------------------------------------
def creer_pretraitement() -> ColumnTransformer:
    """Imputation + mise à l'échelle des nombres, encodage one-hot du protocole."""
    numerique = Pipeline([
        ("imputation", SimpleImputer(strategy="median")),
        ("echelle", StandardScaler()),
    ])
    texte = Pipeline([
        ("imputation", SimpleImputer(strategy="most_frequent")),
        ("encodage", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numerique, COLONNES_NUMERIQUES),
        ("txt", texte, COLONNES_TEXTE),
    ])


def creer_modeles() -> dict:
    """Les 3 modèles comparés, chacun dans un Pipeline avec le prétraitement."""
    modeles = {
        "Régression logistique": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Arbre de décision": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
        "Forêt aléatoire": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    }
    return {nom: Pipeline([("pretraitement", creer_pretraitement()), ("modele", m)])
            for nom, m in modeles.items()}


# ----------------------------------------------------------------------------
# 3. Entraînement et évaluation sur un jeu de validation
# ----------------------------------------------------------------------------
def evaluer(nom: str, pipeline: Pipeline, X_val, y_val) -> dict:
    y_pred = pipeline.predict(X_val)
    vn, fp, fn, vp = confusion_matrix(y_val, y_pred).ravel()
    print(f"\n--- {nom} ---")
    print(classification_report(y_val, y_pred, target_names=["normal", "attaque"], digits=3))
    print(f"Matrice de confusion : VN={vn}  FP={fp}  FN={fn}  VP={vp}")
    return {"modèle": nom,
            "F1 (attaque)": f1_score(y_val, y_pred),
            "Rappel (attaque)": recall_score(y_val, y_pred),
            "Précision (attaque)": precision_score(y_val, y_pred),
            "faux négatifs": int(fn), "faux positifs": int(fp)}


def dessiner_matrice(nom: str, pipeline: Pipeline, X_val, y_val) -> None:
    DOSSIER_FIGURES.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_estimator(pipeline, X_val, y_val, display_labels=["normal", "attaque"],
                                          cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"Matrice de confusion — {nom}")
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")
    fig.tight_layout()
    fig.savefig(DOSSIER_FIGURES / "matrice_confusion.png", dpi=120)
    plt.close(fig)


def analyser_erreurs(pipeline: Pipeline, X_val, y_val) -> None:
    """Affiche les attaques ratées (faux négatifs) et les fausses alertes (faux positifs)."""
    y_pred = pipeline.predict(X_val)
    print("\nAttaques ratées (faux négatifs) :")
    print(X_val[(y_val == 1) & (y_pred == 0)].to_string())
    print("\nFausses alertes (faux positifs) :")
    print(X_val[(y_val == 0) & (y_pred == 1)].to_string())
    print("\nMédianes des attaques détectées, pour comparer :")
    print(X_val[(y_val == 1) & (y_pred == 1)].median(numeric_only=True).round(2).to_string())


# ----------------------------------------------------------------------------
# Programme principal
# ----------------------------------------------------------------------------
def main() -> None:
    df = pd.read_csv(FICHIER_DONNEES)
    explorer(df)

    X = df.drop(columns=[CIBLE])
    y = df[CIBLE]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    print(f"\nJeu d'entraînement : {len(X_train)} flux — jeu de validation : {len(X_val)} flux")

    print("\n=== 2-3. Entraînement et évaluation ===")
    modeles = creer_modeles()
    resultats = []
    for nom, pipeline in modeles.items():
        pipeline.fit(X_train, y_train)
        resultats.append(evaluer(nom, pipeline, X_val, y_val))

    tableau = pd.DataFrame(resultats).set_index("modèle").round(3)
    print("\n=== Comparaison sur le jeu de validation ===")
    print(tableau.to_string())

    print("\n=== Vérification : validation croisée à 5 plis (F1 attaque, moyenne ± écart-type) ===")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for nom, pipeline in creer_modeles().items():
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1")
        print(f"{nom:<22} {scores.mean():.3f} ± {scores.std():.3f}")

    meilleur = tableau["F1 (attaque)"].idxmax()
    print(f"\n=== 4. Modèle retenu : {meilleur} (meilleur F1 de la classe attaque) ===")
    dessiner_matrice(meilleur, modeles[meilleur], X_val, y_val)
    analyser_erreurs(modeles[meilleur], X_val, y_val)

    modele_final = clone(modeles[meilleur]).fit(X, y)      # ré-entraînement sur TOUTES les données
    FICHIER_MODELE.parent.mkdir(exist_ok=True)
    joblib.dump(modele_final, FICHIER_MODELE)
    print(f"=== 5. Modèle enregistré dans {FICHIER_MODELE.relative_to(RACINE)} ===")


if __name__ == "__main__":
    main()
