from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.inprocessing import PrejudiceRemover
import pandas as pd

def apply_fairness(X_train, y_train, X_test, y_test, privileged_attr='Sex'):
    # Convert to AIF360 dataset
    dataset_orig = BinaryLabelDataset(favorable_label=1, unfavorable_label=0,
                                      df=pd.concat([X_train, y_train], axis=1),
                                      label_names=['KTAS'], protected_attribute_names=[privileged_attr])

    dataset_test = BinaryLabelDataset(favorable_label=1, unfavorable_label=0,
                                      df=pd.concat([X_test, y_test], axis=1),
                                      label_names=['KTAS'], protected_attribute_names=[privileged_attr])

    # Train fair model
    model = PrejudiceRemover(sensitive_attr=privileged_attr, eta=1.0)
    model.fit(dataset_orig)
    predictions = model.predict(dataset_test)

    return predictions
