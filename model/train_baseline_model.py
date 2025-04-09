import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv("data/ktas.csv")

# Map categorical encodings
df['Sex'] = df['Sex'].map({1: 'Female', 2: 'Male'})
df['Injury'] = df['Injury'].map({1: 0, 2: 1})
df['Pain'] = df['Pain'].map({0: 0, 1: 1})
df['Mental'] = df['Mental'].map({1: 'Alert', 2: 'Verbal', 3: 'Pain', 4: 'Unresponsive'})

# Drop irrelevant or empty columns (adjust as needed)
df = df.dropna()

# Features & labels
X = df.drop(columns=['KTAS'])
y = df['KTAS'].apply(lambda x: 0 if x >= 4 else 1)  # Emergency (1) vs Non-emergency (0)

# Encode categoricals
X = pd.get_dummies(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model/triage_model.pkl")

# Evaluate
print(classification_report(y_test, model.predict(X_test)))
