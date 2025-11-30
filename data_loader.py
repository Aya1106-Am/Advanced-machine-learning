import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import sys

# Ajouter la racine du projet au chemin Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET_PATH, TEST_SIZE, RANDOM_SEED, PROCESSED_DATA_DIR

class DataLoader:
    def __init__(self, dataset_path=DATASET_PATH):
        self.dataset_path = dataset_path
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.class_names = None

    def load_data(self):
        print(f" Chargement du dataset depuis: {self.dataset_path}")
        try:
            data = pd.read_csv(self.dataset_path, sep=",", header=None)
        except Exception:
            data = pd.read_csv(self.dataset_path, sep=" ", header=None)

        print(f" Dataset chargé: {data.shape[0]} exemples, {data.shape[1]} colonnes")

        # La DERNIÈRE colonne = label texte (AlefI, DadI, DalI, ...)
        y = data.iloc[:, -1].values

        # On veut uniquement des colonnes numériques dans X.
        # Ici on prend toutes les colonnes sauf les 4 dernières
        # (indices / score / label). Ajuste si besoin, mais surtout
        # ne pas inclure la dernière colonne dans X.
        X = data.iloc[:, :-4].values

        print(f" Features shape: {X.shape}")
        print(f" Labels shape: {y.shape}")
        print(f" Nombre de classes uniques: {len(np.unique(y))}")
        return X, y

    def preprocess_data(self, X, y):
        print("\n Prétraitement des données...")
        X_scaled = self.scaler.fit_transform(X)
        y_encoded = self.label_encoder.fit_transform(y)
        self.class_names = self.label_encoder.classes_
        print(" Features normalisées")
        print(f" Labels encodés: {len(self.class_names)} classes")
        return X_scaled, y_encoded

    def split_data(self, X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED):
        print(f"\n Division train/test (test_size={test_size})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        print(f" Train: {X_train.shape[0]} exemples")
        print(f" Test:  {X_test.shape[0]} exemples")
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        return X_train, X_test, y_train, y_test

    def prepare_data(self):
        X, y = self.load_data()
        X, y = self.preprocess_data(X, y)
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        return X_train, X_test, y_train, y_test, self.class_names


def load_arabic_dataset():
    loader = DataLoader()
    return loader.prepare_data()
