import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

def check_gender_fairness(model):
    df = pd.read_csv("data/ktas.csv", encoding='ISO-8859-1', delimiter=';')

    df['KTAS duration_min'] = df['KTAS duration_min'].str.replace(',', '.').astype(float)
    df['Length of stay_min'] = df['Length of stay_min'].astype(float)

    df['Sex'] = df['Sex'].map({1: 'Female', 2: 'Male'})
    df['Injury'] = df['Injury'].map({1: 0, 2: 1})
    df['Pain'] = df['Pain'].map({0: 0, 1: 1})
    df['Mental'] = df['Mental'].map({1: 'Alert', 2: 'Verbal', 3: 'Pain', 4: 'Unresponsive'})

    df = df.dropna()
    df['target'] = df['KTAS_expert'].apply(lambda x: 0 if x >= 4 else 1)

    df_encoded = pd.get_dummies(df.drop(columns=['KTAS_expert']))
    cols = joblib.load("model/feature_columns.pkl")
    df_encoded = df_encoded.reindex(columns=cols, fill_value=0)

    df['prediction'] = model.predict(df_encoded)
    accuracy_by_gender = df.groupby('Sex').apply(lambda x: accuracy_score(x['target'], x['prediction'])).to_dict()

    return {
        "accuracy_by_gender": accuracy_by_gender,
        "difference": abs(accuracy_by_gender['Male'] - accuracy_by_gender['Female'])
    }
