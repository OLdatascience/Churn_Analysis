# Churn_Analysis
Prédire le churn client d’un opérateur télécom et transformer le score en actions de rétention concrètes : choix de seuil (compromis vs rappel prioritaire) et approche Top-k (budget fixe).

# Modèles
Régression Logistique (baseline correcte et interprétable) & Random Forest (performance).
ROC-AUC : LogReg = 0,84 & RandomForest = 0,94.

# Sélection du seuil
- Compromis optimal (F1-max) RegLog ~0.55 & RF ~0,26  (Préc. 0,79 ; Rappel 0,85 ; F1 0,82) 
- Rappel prioritaire (Précision ≥ 60 %) : uniquement RF ~0,17 (Préc. 0,63 ; Rappel 0,89 ; F1 0,74)
- Top-k : cibler les k % de clients à plus forte probabilité (ex. 10 %)
