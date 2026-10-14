# 🧬 NovaGen Research Labs: Health Risk Classification

Predicting whether a person is **healthy** or **unhealthy** from their health records, using supervised machine learning.

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-Data-150458?style=flat&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Models-F7931E?style=flat&logo=scikitlearn&logoColor=white)

> **Result in one line:** the **Random Forest** is the best model. It catches **95.88%** of high-risk people (recall) with **93.77%** accuracy.

---

## 📋 Problem statement

**NovaGen Research Labs** is a biomedical research institute that studies large populations. It wants to learn how a person's underlying health shapes their disease risk and long-term outcomes. Every year, thousands of volunteers go through medical exams, lifestyle assessments and clinical tests.

The researchers are missing one thing: a dependable way to tell people with generally **healthy** profiles apart from people at **higher health risk**. Without it, choosing participants and comparing groups in their studies is much harder.

To solve this, the institute collected **health records for 9,800 people** across several observational studies. There is one record per person, which avoids sampling bias and keeps every observation independent. Each record mixes **numerical and categorical health indicators**: body measurements, lifestyle habits and medical history.

### 🎯 The task

Working as a **data scientist**, build a predictive model that classifies each person as **"healthy"** or **"unhealthy"** from their health data. The institute will use the model to:

- pick eligible participants for clinical trials and long-term (longitudinal) studies
- group people by risk level for analysis and outcome comparison

---

## 📊 The dataset: `novagen_dataset.csv`

| At a glance | |
|---|---|
| Participants (rows) | **9,549** |
| Columns | **23**: 22 health features + `Target` |
| Missing values | None |
| Duplicate rows | None |
| Target balance | `1`: 4,979 people (52.1%) · `0`: 4,570 people (47.9%) |

> The problem statement mentions 9,800 people; the CSV in this repo contains 9,549 rows.

The two classes are almost the same size, so the dataset is well balanced.

### What each column means

**Health measurements**

| Column | Meaning | Values in the file |
|---|---|---|
| `Age` | Age in years | 0 to 100 |
| `BMI` | Body Mass Index (body fat estimated from height and weight) | 19 to 32 |
| `Blood_Pressure` | Systolic blood pressure (mmHg) | 22 to 225 |
| `Cholesterol` | Cholesterol level (mg/dL) | 192 to 207 |
| `Glucose_Level` | Blood glucose level (mg/dL) | 93 to 107 |
| `Heart_Rate` | Resting heart rate (beats per minute) | 67 to 80 |

**Lifestyle**

| Column | Meaning | Values in the file |
|---|---|---|
| `Sleep_Hours` | Average hours of sleep per day | 0 to 14 |
| `Exercise_Hours` | Average hours of exercise per day | 0 to 8 |
| `Water_Intake` | Water drunk per day (litres) | 0 to 10 |
| `Stress_Level` | Stress on a numeric scale (for example 1 to 10) | 0 to 12 |

**Coded categories** (each stored as a number)

| Column | Meaning | Values in the file |
|---|---|---|
| `Smoking` | Smoking habit (described as 1 = smoker, 0 = non-smoker) | 0, 1, 2 |
| `Alcohol` | Alcohol consumption (described as 1 = yes, 0 = no) | 0, 1, 2 |
| `Diet` | General diet category | 0, 1, 2 |
| `MentalHealth` | Mental-health score or condition indicator | 0, 1, 2 |
| `PhysicalActivity` | Overall physical-activity level | 0, 1, 2 |
| `MedicalHistory` | Prior medical conditions | 0, 1, 2 |
| `Allergies` | Known allergies | 0, 1, 2 |

> `Smoking` and `Alcohol` are described as yes/no, but the file stores three codes (0, 1, 2) for both.

**One-hot columns** (`True` / `False`)

| Column | Meaning | `True` in |
|---|---|---|
| `Diet_Type__Vegan` | Diet type is vegan | 3,145 people |
| `Diet_Type__Vegetarian` | Diet type is vegetarian | 3,206 people |
| `Blood_Group_AB` | Blood group AB | 2,339 people |
| `Blood_Group_B` | Blood group B | 2,424 people |
| `Blood_Group_O` | Blood group O | 2,412 people |

A person has at most one `True` in each group. When both diet columns are `False` (3,198 people), the person follows another diet. When all three blood-group columns are `False` (2,374 people), the person is in the remaining group, A.

**Target**

| Column | Meaning | Values in the file |
|---|---|---|
| `Target` | Health outcome or risk, the label the model predicts | 0 or 1 |

The notebook treats **`1` as the higher-risk ("unhealthy") class**: that is the class its recall score measures.

---

## 🧠 The solution, step by step

Everything below comes from [`novagen.ipynb`](novagen.ipynb). The code is copied from the notebook and the numbers are its printed output.

### Step 1: Import the libraries

`pandas` and `numpy` handle the data; scikit-learn provides the train/test split, scaling, metrics and the five models.

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, recall_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
```

### Step 2: Load the data and split it into train and test sets

```python
df = pd.read_csv("novagen_dataset.csv")

# Split features and target
X = df.drop("Target", axis=1)
y = df["Target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

- `X` holds the 22 feature columns; `y` holds the `Target` label.
- 80% of the people (**7,639**) are used for training and 20% (**1,910**) are kept aside for testing.
- `stratify=y` keeps the same healthy/unhealthy ratio in both sets, and `random_state=42` makes the split the same every time.

### Step 3: Scale the features

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

`StandardScaler` puts every feature on the same scale (mean 0, standard deviation 1), so a column with big numbers, like `Blood_Pressure`, doesn't outweigh a column with small numbers, like `Water_Intake`. The scaler **learns only from the training data** and then applies the same transformation to the test data, so no test information leaks into training.

Logistic Regression, KNN and the Voting Classifier use the scaled data. Random Forest and Gradient Boosting are trained on the original values.

### Step 4: Pick the right metric (recall)

The notebook's reasoning:

```python
# In Model Evaluation, Recall is more important than accuracy 
# because missing a high-risk patient is dangerous
```

**Recall** answers: *of all the people who really are high-risk, how many did the model catch?* A model with low recall would label some at-risk people as healthy, which is the costly mistake here. So each model is scored on accuracy **and** recall, and recall decides the winner. Every model is evaluated with the same three lines, for example:

```python
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Logistic Regression Recall:", recall_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))
```

### Step 5: Train the five models

#### Model 1: Logistic Regression (baseline)

```python
log_reg = LogisticRegression(
    penalty="l2",
    solver="liblinear",
    max_iter=1000
)

log_reg.fit(X_train_scaled, y_train)

y_pred_lr = log_reg.predict(X_test_scaled)
```

A simple linear model that estimates the chance that a person is high-risk. `penalty="l2"` adds regularization, which keeps the model's weights small so it doesn't overfit.

**Accuracy 81.41% · Recall 82.83%**

#### Model 2: K-Nearest Neighbours (KNN)

```python
knn = KNeighborsClassifier(
    n_neighbors=5,
    metric="euclidean"
)

knn.fit(X_train_scaled, y_train)

y_pred_knn = knn.predict(X_test_scaled)
```

To classify a person, KNN finds the **5 most similar people** in the training data, measured by straight-line (Euclidean) distance, and goes with the majority label. Because it compares distances, it uses the scaled features.

**Accuracy 88.32% · Recall 88.35%**

#### Model 3: Random Forest (ensemble)

```python
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
```

A Random Forest builds **200 decision trees**, each on a random sample of the data, and lets them vote. `max_depth=None` lets every tree grow fully. It is trained on the original, unscaled features.

**Accuracy 93.77% · Recall 95.88%** 🏆

#### Model 4: Gradient Boosting (ensemble)

```python
gb = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gb.fit(X_train, y_train)

y_pred_gb = gb.predict(X_test)
```

Gradient Boosting builds **150 small trees** (depth 3) one after another, and each new tree fixes the mistakes of the ones before it. `learning_rate=0.1` sets how much each new tree is allowed to change the prediction.

**Accuracy 93.04% · Recall 94.98%**

#### Model 5: Voting Classifier (ensemble)

```python
voting_clf = VotingClassifier(
    estimators=[
        ("lr", LogisticRegression(max_iter=1000, solver="liblinear")),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42))
    ],
    voting="soft"
)

voting_clf.fit(X_train_scaled, y_train)

y_pred_vote = voting_clf.predict(X_test_scaled)
```

This combines three models: Logistic Regression, KNN and a Random Forest. With `voting="soft"`, it averages their predicted probabilities and picks the class with the higher average.

**Accuracy 91.62% · Recall 93.07%**

---

## 📈 Results

Tested on the 1,910 held-out people (914 labelled `0`, 996 labelled `1`):

| Model | Accuracy | Recall (high-risk class) |
|---|:---:|:---:|
| Logistic Regression | 81.41% | 82.83% |
| KNN | 88.32% | 88.35% |
| **Random Forest** 🏆 | **93.77%** | **95.88%** |
| Gradient Boosting | 93.04% | 94.98% |
| Voting Classifier | 91.62% | 93.07% |

Full classification report for the Random Forest:

```
              precision    recall  f1-score   support

           0       0.95      0.91      0.93       914
           1       0.92      0.96      0.94       996

    accuracy                           0.94      1910
   macro avg       0.94      0.94      0.94      1910
weighted avg       0.94      0.94      0.94      1910
```

## ✅ Conclusion

Based on **recall**, the model NovaGen should use is the **Random Forest**. Of the 996 high-risk people in the test set, it correctly flags **955 (95.88%)**, and it is right about **93.77%** of all test cases.

---

## 🗂️ Repository structure

```
NovaGen_Lab_Project/
├── novagen.ipynb         # the full solution notebook
├── novagen_dataset.csv   # the dataset (9,549 participants)
└── README.md
```

## ▶️ How to run it

```bash
git clone https://github.com/Preetbbhalala/NovaGen_Lab_Project.git
cd NovaGen_Lab_Project
pip install pandas numpy scikit-learn jupyter
jupyter notebook novagen.ipynb
```

Then choose **Run All**. The notebook reads `novagen_dataset.csv` from the same folder.

---

## 👤 Author

**Preet Bhalala**: [@Preetbbhalala](https://github.com/Preetbbhalala)

## ⭐ Like this project?

If you found it useful or interesting, please give the repo a **star** ⭐. It really helps and is much appreciated!
