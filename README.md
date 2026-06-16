# House Price Worth-it Classifier for Gen Z in Surabaya

🔗 **[Open Project in Google Colab](https://drive.google.com/file/d/1HoRob39mlsY3sz9JRUxuULKb8BhSdWLN/view?usp=sharing)**

## 📌 Project Overview
This repository contains the official implementation of our **PKM-AI (Student Scientific Article)** project titled: 

> **"KLASIFIKASI TINGKAT KELAYAKAN HARGA RUMAH MENGGUNAKAN MACHINE LEARNING BERBASIS FITUR PROPERTI DAN LOKASI UNTUK GENERASI Z DI SURABAYA"**

Rising property prices in Surabaya up to 2025 have posed a significant challenge for Generation Z in determining whether a house's listing price is objectively "worth-it" relative to its offered features and location. This project addresses the issue by developing a data-driven Machine Learning classification framework to evaluate house price fairness objectively, consistently, and transparently, serving as an intelligent decision support system.

---

## 📊 Dataset & Preprocessing
* **Data Source:** 248 landed house samples collected through **automated web scraping** in Surabaya.
* **Target Labeling:** Binary classification (`Worth-it` vs `Not Worth-it`) annotated by a panel of Gen Z respondents using a majority voting mechanism.
* **Key Features:** Property specifications (building size, land area, bedrooms, bathrooms) and location metrics.
* **Dataset File:** Saved as `Survey_Combined_MajorityVote.csv`.

---

## ⚙️ Methodology & Pipeline
We implemented an end-to-end Machine Learning pipeline handled by Scikit-Learn's `ColumnTransformer`. The pipeline automatically executes preprocessing steps (scaling numeric variables and encoding categorical features) internally to prevent data leakage during evaluation.

Four distinct algorithms were trained and rigorously evaluated using **Stratified 5-Fold Cross-Validation**:
1. Logistic Regression
2. Random Forest
3. Decision Tree *(Best Performer)*
4. Support Vector Machine (SVM)

---

## 🏆 Evaluation & Results
Based on our experimental benchmarks (exported in `model_comparison2.csv`), the **Decision Tree Classifier** emerged as the most stable and optimal model for this task.

* **Best Model:** Decision Tree Classifier
* **Cross-Validation F1-Score:** **0.8114**
* **Primary Predictor:** Listing Price (*Harga Jual*) was identified as the most statistically significant variable affecting the worth-it score classification.

---

## 📦 Saved Artifacts (`/models`)
To ensure reproducibility and ease of deployment, all required inference components are modularly saved inside the `models/` directory:
* `best_model2.pkl`: The complete trained Decision Tree pipeline (includes `ColumnTransformer` preprocessing + classifier).
* `label_encoder2.pkl`: Decodes model outputs back to human-readable text (`0` / `1` -> `Not Worth-it` / `Worth-it`).
* `feature_columns2.pkl`: Stores the strict feature schema and column order required during inference.
* `model_metadata2.pkl`: Dictionary capturing evaluation metrics and training configurations.
