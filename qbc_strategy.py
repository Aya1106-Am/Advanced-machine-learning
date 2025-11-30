"""
Stratégie Query-by-Committee (QBC)
Sélectionne les exemples où le comité de modèles est le plus en désaccord
"""

import numpy as np
from scipy.stats import entropy


class QBCQueryStrategy:
    """
    Query-by-Committee: sélection basée sur le désaccord entre modèles.
    
    Cette stratégie entraîne plusieurs modèles (le "comité") et sélectionne
    les exemples pour lesquels les modèles prédisent des classes différentes.
    
    Le désaccord est mesuré par plusieurs métriques combinées.
    """
    
    def __init__(self):
        self.name = "QBC"
    
    def select(self, models, X_unlabeled, n_instances):
        """
        Sélectionne les échantillons où le comité est le plus en désaccord.
        
        Args:
            models: liste de ModelWrapper ou modèles sklearn déjà entraînés (le comité)
            X_unlabeled: données non étiquetées (shape: n_samples, n_features)
            n_instances: nombre d'échantillons à sélectionner
        
        Returns:
            indices: array des indices des échantillons sélectionnés
        """
        if n_instances <= 0:
            return np.array([], dtype=int)
        
        if n_instances > len(X_unlabeled):
            n_instances = len(X_unlabeled)
        
        if not models or len(models) == 0:
            raise ValueError("Le comité doit contenir au moins un modèle")
        
        # Obtenir les prédictions de chaque modèle du comité
        all_probs = []
        all_classes_seen = set()
        
        for model in models:
            # Gérer ModelWrapper et modèles sklearn directs
            if hasattr(model, 'model'):
                # C'est un ModelWrapper
                actual_model = model.model
                probs = actual_model.predict_proba(X_unlabeled)
                classes = actual_model.classes_
            elif hasattr(model, 'predict_proba'):
                # C'est un modèle sklearn direct
                probs = model.predict_proba(X_unlabeled)
                classes = model.classes_
            else:
                raise ValueError("Les modèles du comité doivent avoir une méthode predict_proba")
            
            all_probs.append((probs, classes))
            all_classes_seen.update(classes)
        
        # Convertir en liste triée pour avoir un ordre cohérent
        all_classes = sorted(list(all_classes_seen))
        n_classes = len(all_classes)
        n_samples = len(X_unlabeled)
        n_models = len(models)
        
        # Créer une matrice normalisée (n_models, n_samples, n_classes)
        normalized_probs = np.zeros((n_models, n_samples, n_classes))
        
        for model_idx, (probs, classes) in enumerate(all_probs):
            # Créer un mapping des classes du modèle vers l'index global
            class_mapping = {cls: all_classes.index(cls) for cls in classes}
            
            # Remplir la matrice normalisée
            for sample_idx in range(n_samples):
                for cls_idx, cls in enumerate(classes):
                    global_cls_idx = class_mapping[cls]
                    normalized_probs[model_idx, sample_idx, global_cls_idx] = probs[sample_idx, cls_idx]
        
        # Maintenant tous les modèles ont la même shape, on peut calculer le désaccord
        
        # Méthode 1: Variance des probabilités
        variance_disagreement = np.var(normalized_probs, axis=0).sum(axis=1)
        
        # Méthode 2: Vote Entropy - regarder les prédictions de classe
        predictions = np.argmax(normalized_probs, axis=2)  # (n_models, n_samples)
        vote_entropy_scores = np.zeros(n_samples)
        
        for sample_idx in range(n_samples):
            votes = predictions[:, sample_idx]
            unique_votes, counts = np.unique(votes, return_counts=True)
            vote_probs = counts / len(votes)
            vote_entropy_scores[sample_idx] = entropy(vote_probs)
        
        # Méthode 3: Consensus Entropy - entropie de la distribution moyenne
        mean_probs = np.mean(normalized_probs, axis=0)  # (n_samples, n_classes)
        consensus_entropy = entropy(mean_probs.T)
        
        # Normaliser chaque métrique
        def safe_normalize(arr):
            max_val = np.max(arr)
            if max_val > 1e-10:
                return arr / max_val
            return arr
        
        variance_disagreement = safe_normalize(variance_disagreement)
        vote_entropy_scores = safe_normalize(vote_entropy_scores)
        consensus_entropy = safe_normalize(consensus_entropy)
        
        # Combiner les trois métriques
        disagreement = (
            0.4 * variance_disagreement +
            0.4 * vote_entropy_scores +
            0.2 * consensus_entropy
        )
        
        # Sélectionner les indices avec le plus de désaccord
        query_idx = np.argsort(-disagreement)[:n_instances]
        
        return query_idx
    
    def __repr__(self):
        return f"QBCQueryStrategy(name='{self.name}')"