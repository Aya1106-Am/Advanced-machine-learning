"""
Boucle principale d'Active Learning (stratégies: Entropy, QBC, Margin, etc.)
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import os
import sys

# Ajouter la racine du projet au chemin Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import INITIAL_LABELED_SIZE, QUERY_BATCH_SIZE, N_ITERATIONS
from config import MODEL_TYPE, N_COMMITTEE
from src.models import ModelWrapper
from src.query_strategies import get_query_strategy


class ActiveLearner:
    def __init__(self, X_train, y_train, X_test, y_test,
                 strategy_name="entropy", model_type=MODEL_TYPE,
                 n_committee=N_COMMITTEE):
        self.X_pool = X_train.copy()
        self.y_pool = y_train.copy()
        self.X_test = X_test
        self.y_test = y_test

        self.strategy_name = strategy_name
        self.strategy = get_query_strategy(strategy_name)

        self.model_type = model_type
        self.model = ModelWrapper(model_type)

        # Comité pour QBC (on stocke des ModelWrapper, pas des modèles bruts)
        self.n_committee = n_committee
        self.committee = None
        if strategy_name == "QBC":
            # Créer des modèles avec des random_state différents pour plus de diversité
            self.committee = []
            for i in range(n_committee):
                # Utiliser un random_state différent pour chaque modèle du comité
                wrapper = ModelWrapper(model_type)
                # Modifier le random_state du modèle pour avoir de la diversité
                if hasattr(wrapper.model, 'random_state'):
                    wrapper.model.set_params(random_state=42 + i)
                self.committee.append(wrapper)

        self.X_labeled = None
        self.y_labeled = None
        self.labeled_indices = []
        self.unlabeled_indices = list(range(len(X_train)))

        self.history = {
            "n_labeled": [],
            "accuracy": [],
            "f1_macro": [],
            "f1_micro": []
        }

        print("=" * 60)
        print(f" Active Learner")
        print(f"  - Stratégie: {self.strategy_name}")
        print(f"  - Modèle: {model_type}")
        if strategy_name == "QBC":
            print(f"  - Taille du comité: {n_committee}")
            print(f"  - Random states du comité: {[42+i for i in range(n_committee)]}")
        print(f"  - Pool: {len(self.X_pool)} | Test: {len(self.X_test)}")
        print("=" * 60)

    def initialize_labeled_set(self, n_initial=INITIAL_LABELED_SIZE):
        print(f"\n Initialisation avec {n_initial} exemples étiquetés aléatoirement...")
        initial_indices = np.random.choice(
            self.unlabeled_indices,
            size=n_initial,
            replace=False
        )
        self.X_labeled = self.X_pool[initial_indices]
        self.y_labeled = self.y_pool[initial_indices]
        self.labeled_indices = list(initial_indices)
        self.unlabeled_indices = [i for i in self.unlabeled_indices if i not in initial_indices]
        print(f"  Labeled: {len(self.labeled_indices)} | Unlabeled: {len(self.unlabeled_indices)}")

    def train_model(self):
        """Entraîne le modèle principal"""
        self.model.fit(self.X_labeled, self.y_labeled)

    def train_committee(self):
        """
        Entraîne chaque modèle du comité sur un sous-échantillon bootstrap.
        Chaque modèle voit un échantillon différent pour créer de la diversité.
        """
        n_samples = len(self.X_labeled)
        
        for idx, model_wrapper in enumerate(self.committee):
            # Bootstrap avec remplacement (méthode standard)
            # Chaque modèle voit environ 63% d'exemples uniques
            bootstrap_indices = np.random.choice(
                n_samples,
                size=n_samples,
                replace=True
            )
            
            # Vérifier que toutes les classes sont présentes
            unique_classes_total = np.unique(self.y_labeled)
            unique_classes_bootstrap = np.unique(self.y_labeled[bootstrap_indices])
            
            # Si des classes manquent, forcer leur inclusion
            missing_classes = set(unique_classes_total) - set(unique_classes_bootstrap)
            if missing_classes:
                additional_indices = []
                for cls in missing_classes:
                    cls_indices = np.where(self.y_labeled == cls)[0]
                    if len(cls_indices) > 0:
                        # Ajouter au moins 2 exemples de chaque classe manquante
                        n_add = min(2, len(cls_indices))
                        additional_indices.extend(
                            np.random.choice(cls_indices, size=n_add, replace=False)
                        )
                
                # Remplacer les derniers éléments par les classes manquantes
                if additional_indices:
                    bootstrap_indices = np.concatenate([
                        bootstrap_indices[:-len(additional_indices)],
                        additional_indices
                    ])
            
            # Entraîner le modèle
            X_bootstrap = self.X_labeled[bootstrap_indices]
            y_bootstrap = self.y_labeled[bootstrap_indices]
            
            model_wrapper.fit(X_bootstrap, y_bootstrap)

    def evaluate_model(self):
        """Évalue le modèle principal sur l'ensemble de test"""
        y_pred = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        f1_macro = f1_score(self.y_test, y_pred, average="macro")
        f1_micro = f1_score(self.y_test, y_pred, average="micro")
        return acc, f1_macro, f1_micro

    def query_instances(self, n_instances=QUERY_BATCH_SIZE):
        """Sélectionne les instances à annoter selon la stratégie"""
        X_unlabeled = self.X_pool[self.unlabeled_indices]
        
        # Adapter l'input selon la stratégie
        if self.strategy_name == "QBC":
            model_input = self.committee
        elif self.strategy_name == "random":
            model_input = None  # Random sampling n'a pas besoin du modèle
        else:
            model_input = self.model.model
        
        # Sélection
        local_idx = self.strategy.select(
            model_input,
            X_unlabeled,
            min(n_instances, len(self.unlabeled_indices))
        )
        
        # Convertir les indices locaux en indices globaux
        selected = [self.unlabeled_indices[i] for i in local_idx]
        return selected

    def label_instances(self, indices):
        """Ajoute les instances sélectionnées à l'ensemble étiqueté"""
        new_X = self.X_pool[indices]
        new_y = self.y_pool[indices]
        self.X_labeled = np.vstack([self.X_labeled, new_X])
        self.y_labeled = np.concatenate([self.y_labeled, new_y])
        self.labeled_indices.extend(indices)
        self.unlabeled_indices = [i for i in self.unlabeled_indices if i not in indices]

    def run(self, n_iterations=N_ITERATIONS, n_instances_per_iteration=QUERY_BATCH_SIZE):
        """Boucle principale de l'Active Learning"""
        self.initialize_labeled_set()

        for it in range(n_iterations):
            if len(self.unlabeled_indices) == 0:
                print(f"\n Plus d'exemples non étiquetés (itération {it})")
                break

            print(f"\n--- Itération {it + 1}/{n_iterations} ---")
            print(f"  Labeled: {len(self.labeled_indices)} | Unlabeled: {len(self.unlabeled_indices)}")

            # Entraînement
            # Toujours entraîner le modèle principal en premier
            self.train_model()
            
            # Pour QBC, entraîner aussi le comité
            if self.strategy_name == "QBC":
                self.train_committee()

            # Évaluation
            acc, f1_macro, f1_micro = self.evaluate_model()
            self.history["n_labeled"].append(len(self.labeled_indices))
            self.history["accuracy"].append(acc)
            self.history["f1_macro"].append(f1_macro)
            self.history["f1_micro"].append(f1_micro)

            print(f"  Accuracy={acc:.4f} | F1-macro={f1_macro:.4f} | F1-micro={f1_micro:.4f}")

            # Sélection et annotation
            selected = self.query_instances(n_instances_per_iteration)
            self.label_instances(selected)

        print("\n" + "="*60)
        print(" FIN ACTIVE LEARNING")
        print(f"  Accuracy finale: {self.history['accuracy'][-1]:.4f}")
        print(f"  F1-macro final:  {self.history['f1_macro'][-1]:.4f}")
        print(f"  Total exemples:  {self.history['n_labeled'][-1]}")
        print("="*60)
        
        return self.history