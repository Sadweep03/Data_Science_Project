# **Classification of Liver Fibrosis Stages in Hepatitis C Patients Using Clinical and Demographic Data.**

# **Loading Dataset and Files**
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the main dataset and the discretization criteria
df = pd.read_csv('HCV-Egy-Data.csv')
criteria = pd.read_csv('Discretization-Criteria.csv')

# Display basic information
print("Dataset Shape:", df.shape)
print("\nColumn Names and Data Types:")
df.info()

# Display the first few rows
print("\nFirst 5 Rows of the Dataset:")
df.head()

"""# **EDA**

## **Missing Values and Data Quality Check**
"""

# Check for missing values in each column
missing_values = df.isnull().sum()
print("Missing values per column:")
print(missing_values[missing_values > 0]) # Shows only columns with missing data

# Summary statistics for numerical columns
summary_stats = df.describe().T
print("\nSummary Statistics:")
print(summary_stats)

"""## **Target Variable Analysis**"""

# Distribution of the target variable: Liver Fibrosis Stages (1 to 4)
plt.figure(figsize=(8, 5))
sns.countplot(x='Baselinehistological staging', data=df, palette='viridis')
plt.title('Distribution of Liver Fibrosis Stages (Target Variable)')
plt.xlabel('Fibrosis Stage (1=Portal Fibrosis, 2=Few Septa, 3=Many Septa, 4=Cirrhosis)')
plt.ylabel('Count')
plt.savefig('target_distribution.png')
plt.show()

print("Stage Counts:")
print(df['Baselinehistological staging'].value_counts().sort_index())

"""## **Correlation Analysis**"""

# Correlation Heatmap
plt.figure(figsize=(16, 12))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Matrix Heatmap')
plt.savefig('correlation_matrix.png')
plt.show()

# Display correlations specifically with the target variable
print("\nCorrelation with Target (Baselinehistological staging):")
print(correlation_matrix['Baselinehistological staging'].sort_values(ascending=False))

# Define the target variable
target = 'Baselinehistological staging'

# Calculate absolute correlations with the target and sort them
correlations = df.corr()[target].abs().sort_values(ascending=False)

# Select top 10 features (excluding the target itself)
top_10_features = correlations.index[1:11].tolist()

# Create a subset including the target
subset_cols = top_10_features + [target]
correlation_matrix = df[subset_cols].corr()

# Visualize with a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title(f'Correlation Matrix: Top 10 Features vs {target}')
plt.savefig('top_10_correlation_matrix.png')
plt.show()

"""## **Continuous Variable Distributions**"""

continuous_cols = ['Age', 'BMI', 'WBC', 'RBC', 'HGB', 'Plat']

# Clean column names by stripping whitespace
df.columns = df.columns.str.strip()

plt.figure(figsize=(15, 10))
for i, col in enumerate(continuous_cols, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[col], kde=True, color='skyblue')
    plt.title(f'Distribution of {col}')

plt.tight_layout()
plt.savefig('continuous_distributions.png')
plt.show()

"""## **Categorical Variable Analysis**"""

# Bar plots for symptom/categorical variables
categorical_cols = ['Gender', 'Fever', 'Nausea/Vomting', 'Headache', 'Diarrhea', 'Jaundice']

plt.figure(figsize=(15, 10))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(2, 3, i)
    sns.countplot(x=col, data=df, palette='pastel', hue=col, legend=False)
    plt.title(f'Count of {col}')

plt.tight_layout()
plt.savefig('categorical_distributions.png')
plt.show()

df.head()

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. Cleaning Column Names
df.columns = df.columns.str.strip().str.replace('/', '_').str.replace(' ', '_')

# 2. Selecting relevant predictors (Demographics + Serum Labs)
# We exclude "future" RNA values to predict "Baseline" staging
features = ['Age', 'Gender', 'BMI', 'Fever', 'Nausea_Vomting', 'Headache',
            'Diarrhea', 'Fatigue_&_generalized_bone_ache', 'Jaundice',
            'Epigastric_pain', 'ALT_36', 'RNA_Base']
target = 'Baselinehistological_staging'

X = df[features].copy()
y = df[target].copy()

# 3. Encoding & Scaling
# Convert Gender to numeric if it's text
if X['Gender'].dtype == 'object':
    X['Gender'] = LabelEncoder().fit_transform(X['Gender'])

# XGBoost requires labels to start at 0 (your staging likely starts at 1)
y_encoded = y - y.min()

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Initialize Models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, kernel='rbf', random_state=42),
    "XGBoost": XGBClassifier(eval_metric='mlogloss', random_state=42)
}

# Dictionary to store results
results = {}

for name, model in models.items():
    # Training Time
    start_train = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_train

    # Testing Time
    start_test = time.time()
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)
    test_time = time.time() - start_test

    results[name] = {
        "pred": y_pred,
        "prob": y_prob,
        "train_time": train_time,
        "test_time": test_time
    }

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

metrics_list = []

for name, res in results.items():
    metrics_list.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, res['pred']),
        "F1-Score": f1_score(y_test, res['pred'], average='weighted'),
        "Precision": precision_score(y_test, res['pred'], average='weighted'),
        "Recall": recall_score(y_test, res['pred'], average='weighted'),
        "Train Time (s)": res['train_time'],
        "Test Time (s)": res['test_time']
    })

perf_df = pd.DataFrame(metrics_list).set_index("Model")
print(perf_df)

# Print detailed classification reports
for name in models.keys():
    print(f"\n--- Detailed Report: {name} ---")
    print(classification_report(y_test, results[name]['pred']))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# 1. Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
for i, (name, res) in enumerate(results.items()):
    cm = confusion_matrix(y_test, res['pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', ax=axes[i])
    axes[i].set_title(f'Confusion Matrix: {name}')
    axes[i].set_xlabel('Predicted Stage')
    axes[i].set_ylabel('True Stage')

plt.tight_layout()
plt.show()

# 2. ROC Curves (One-vs-Rest for multiclass)
y_test_bin = label_binarize(y_test, classes=np.unique(y_encoded))
n_classes = y_test_bin.shape[1]

plt.figure(figsize=(10, 7))
for name, res in results.items():
    # Compute ROC for each model (Macro-average)
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), res['prob'].ravel())
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Combined')
plt.legend(loc='lower right')
plt.show()

import matplotlib.pyplot as plt

# Using Random Forest to see Feature Importance
rf_model = models["Random Forest"]
importances = rf_model.feature_importances_
feat_names = features
indices = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.title("Feature Importances for Hepatitis Staging")
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [feat_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()

from sklearn.model_selection import GridSearchCV

# Define a search grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
                           param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)

print("Starting Grid Search... this may take a moment.")
grid_search.fit(X_train_scaled, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")

# Evaluate the tuned model
best_rf = grid_search.best_estimator_
tuned_pred = best_rf.predict(X_test_scaled)
print("\nUpdated Classification Report (Tuned RF):")
print(classification_report(y_test, tuned_pred))
