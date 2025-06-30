# 🧠 Détection et Classification des Tumeurs Cérébrales par IA

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-green)

## 📌 Objectif du projet
Système complet d'aide au diagnostic médical capable de :
1. **Détecter** la présence de tumeurs cérébrales sur des IRM
2. **Classifier** le type de tumeur (gliome, méningiome, hypophysaire)
3. **Fournir une interface** clinique intuitive

## 🔍 Méthodologie complète

### 1. Collecte des données
- **Source** : Kaggle (Brain Tumor MRI Dataset)
- **Répartition** :
  - 1,621 gliomes
  - 1,645 méningiomes
  - 1,757 tumeurs hypophysaires
  - 2,000 images saines

### 2. Prétraitement avancé
- Recadrage automatique du cerveau (OpenCV)
- Normalisation (Min-Max scaling)
- Augmentation des données :
  - Rotation (±15°)
  - Zoom (jusqu'à 20%)
  - Retournement horizontal

### 3. Modèles implémentés

#### 🔬 Classification Binaire (Tumeur vs Sain)
| Modèle | Architecture | Accuracy | Precision | Recall | F1-Score | Temps Inférence |
|--------|--------------|----------|-----------|--------|----------|-----------------|
| CNN Custom | 4 Conv + 2 Dense | 96% | 96% | 95% | 96% | 120ms |
| ResNet101 | Transfer Learning | 81% | 81% | 72% | 74% | 180ms |
| EfficientNetB0 | Transfer Learning | 84% | 84% | 76% | 79% | 150ms |
| SVM (RBF Kernel) | - | 94% | 94% | 91% | 93% | 70ms |
| XGBoost | 100 estimators | 94% | 94% | 90% | 92% | 60ms |
| Random Forest | 100 arbres | 86% | 85% | 80% | 82% | 80ms |

#### 🏷 Classification Multi-Classe (4 types)
| Modèle | Accuracy | Precision (Moy) | Recall (Moy) | F1-Score (Moy) |
|--------|----------|-----------------|--------------|----------------|
| EfficientNetB0 | 99.3% | 99% | 99% | 99% |
| VGG16 | 88% | 89% | 88% | 88% |
| ResNet50 | 75% | 74% | 75% | 73% |

**Détail EfficientNetB0** :
| Classe | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| Gliome | 98.7% | 99.2% | 98.9% |
| Méningiome | 99.1% | 98.5% | 98.8% |
| Hypophyse | 99.3% | 99.0% | 99.1% |
| Sain | 99.5% | 99.8% | 99.6% |

### 4. Optimisations
- **Fine-tuning** : Réglage des 100 dernières couches d'EfficientNet
- **Régularisation** : Dropout (0.5), EarlyStopping
- **Augmentation** : Génération de 3 variations par image

## 🛠 Stack Technique
- **ML/DL** : TensorFlow, Keras, scikit-learn
- **Traitement d'images** : OpenCV, Albumentations
- **Visualisation** : Matplotlib, Plotly
- **Interface** : Streamlit
- **Gestion données** : Pandas, NumPy

## 🚀 Déploiement
```bash
# Cloner le dépôt
git clone https://github.com/votre_user/brain-tumor-ai.git
cd brain-tumor-ai

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run app.py
