"""
Stratégie Least Confidence
Sélectionne les exemples où le modèle est le moins confiant
"""

import numpy as np


class LeastConfidenceQueryStrategy:
    """
    Stratégie basée sur la confiance minimale.
    
    Sélectionne les exemples pour lesquels la probabilité maximale
    (confiance) est la plus faible.
    
    Formule: confidence = max(p_i)
    On sélectionne les exemples avec min(confidence)
    """
    
    def __init__(self):
        self.name = "least_confidence"
    
    def select(self, model, X_unlabeled, n_instances):
        """
        Sélectionne les échantillons où le modèle est le moins confiant.
        
        Args:
            model: modèle entraîné avec méthode predict_proba
            X_unlabeled: données non étiquetées (shape: n_samples, n_features)
            n_instances: nombre d'échantillons à sélectionner
        
        Returns:
            indices: array des indices des échantillons sélectionnés
        
        Example:
            >>> strategy = LeastConfidenceQueryStrategy()
            >>> indices = strategy.select(model, X_unlabeled, 10)
        """
        if n_instances <= 0:
            return np.array([], dtype=int)
        
        if n_instances > len(X_unlabeled):
            n_instances = len(X_unlabeled)
        
        # Obtenir les probabilités de prédiction
        probas = model.predict_proba(X_unlabeled)
        
        # Confiance = probabilité maximale pour chaque exemple
        confidences = np.max(probas, axis=1)
        
        # Sélectionner les indices avec les plus faibles confiances
        indices = np.argsort(confidences)[:n_instances]
        
        return indices
    
    def __repr__(self):
        return f"LeastConfidenceQueryStrategy(name='{self.name}')"