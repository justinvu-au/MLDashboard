import streamlit as st

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from joblib import load, dump

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt

from lime import lime_tabular

# Load data
wine = load_wine()
wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
wine_df["WineType"] = [wine.target_names[typ] for typ in wine.target]

X_train, X_test, Y_train, Y_test = train_test_split(wine.data, wine.target, train_size=0.8, random_state=123)
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

## Train model
rf_classif = RandomForestClassifier()
rf_classif.fit(X_train, Y_train)

## Make prediction
Y_test_preds = rf_classif.predict(X_test)
Y_test_preds[:5]

## Evaluate Metrics
print("Test accuracy : {:.2f}".format(accuracy_score(Y_test, Y_test_preds)))

print("\nConfusion Matrix :")
print(confusion_matrix(Y_test, Y_test_preds))

print("\nClassification Report: ")
print(classification_report(Y_test, Y_test_preds))

## Save Model
dump(rf_classif, "rf_classif.model")

rf_classif_2 = load("rf_classif.model")
rf_classif_2

Y_test_preds = rf_classif_2.predict(X_test)

print("Test accuracy : {:.2f}".format(accuracy_score(Y_test, Y_test_preds)))

print("\nConfusion Matrix :")
print(confusion_matrix(Y_test, Y_test_preds))

print("\nClassification Report: ")
print(classification_report(Y_test, Y_test_preds))

## Interpret Model Performance
explainer = lime_tabular.LimeTabularExplainer(X_train, mode="classification", class_names=wine.target_names, feature_names=wine.feature_names)
explainer
explanation = explainer.explain_instance(X_test[0], rf_classif.predict_proba, num_features=len(wine.feature_names), top_labels=3)
explanation.show_in_notebook()
fig = explanation.as_pyplot_figure(label=2)

