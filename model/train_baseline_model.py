import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib


df = pd.read_csv("data/ktas.csv", encoding='ISO-8859-1', delimiter=';')

df['KTAS duration_min'] = df['KTAS duration_min'].str.replace(',', '.').astype(float)
df['Length of stay_min'] = df['Length of stay_min'].astype(float)

df['Sex'] = df['Sex'].map({1: 'Female', 2: 'Male'})
df['Injury'] = df['Injury'].map({1: 0, 2: 1})
df['Pain'] = df['Pain'].map({0: 0, 1: 1})
df['Mental'] = df['Mental'].map({1: 'Alert', 2: 'Verbal', 3: 'Pain', 4: 'Unresponsive'})

df = df.dropna()

X = df.drop(columns=['KTAS_expert'])
y = df['KTAS_expert'].apply(lambda x: 0 if x >= 4 else 1)

X = pd.get_dummies(X)

joblib.dump(X.columns.tolist(), "model/feature_columns.pkl")

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "model/triage_model.pkl")

print(classification_report(y_test, model.predict(X_test)))
