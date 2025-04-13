import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv("data/ktas.csv", encoding='ISO-8859-1', delimiter=';')

# Convert comma decimals to float
df['KTAS duration_min'] = df['KTAS duration_min'].str.replace(',', '.').astype(float)
df['Length of stay_min'] = df['Length of stay_min'].astype(float)

# Encode categorical values
df['Sex'] = df['Sex'].map({1: 'Female', 2: 'Male'})
df['Injury'] = df['Injury'].map({1: 0, 2: 1})
df['Pain'] = df['Pain'].map({0: 0, 1: 1})
df['Mental'] = df['Mental'].map({1: 'Alert', 2: 'Verbal', 3: 'Pain', 4: 'Unresponsive'})

# Drop missing data
df = df.dropna()

# Define features and target
X = df.drop(columns=['KTAS_expert'])
y = df['KTAS_expert'].apply(lambda x: 0 if x >= 4 else 1)

# One-hot encode
X = pd.get_dummies(X)

# Save column names
joblib.dump(X.columns.tolist(), "model/feature_columns.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model/triage_model.pkl")

# Evaluation
print(classification_report(y_test, model.predict(X_test)))
