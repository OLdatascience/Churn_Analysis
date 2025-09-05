# Churn_Analysis
Prédire le churn client d’un opérateur télécom et transformer le score en actions de rétention concrètes : **choix de seuil** (compromis vs rappel prioritaire) et **approche Top-k** (budget fixe).

# Modèles
**Régression Logistique** (baseline correcte et interprétable) - AUC **0.84**
**Random Forest** (performance) - AUC **0.94**

# Sélection du seuil
- **Compromis optimal (F1-max)**
  - RegLog seuil ~ **0.55**  
  - Random Forest ~ **0.26**  avec Précision **0.79**, Rappel **0.85** & F1 **0.82** 
- **Si le rappel est prioritaire** (avec précision ≥ 60 %) : RF ~ **0.17** , Préc. **0.63** , Rappel **0.89** & F1 **0.74**
- **Top-k** : cibler les k % de clients à plus forte probabilité (ex. 10 %)

# Installation (au choix) 
# Windows PowerShell
python -m venv .venv 
.\.venv\Scripts\Activate.ps1 
pip install -r requirements.txt

# MacOS/Linux
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Conda
conda create -n churn python=3.11 -y
conda activate churn 
pip install -r requirements.txt

## Exécution
python src/Churn_analysis.py
