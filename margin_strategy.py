"""
Stratégie Margin Sampling
Sélectionne les exemples avec la plus petite marge entre les deux meilleures prédictions
"""

import numpy as np


class MarginSamplingQueryStrategy:
    """
    Stratégie basée sur la marge entre les deux meilleures prédictions.
    
    La marge mesure la différence entre les probabilités des deux classes
    les plus probables. Une petite marge indique que le modèle hésite
    entre deux classes.
    
    Formule: margin = p_1 - p_2
    où p_1 et p_2 sont les deux plus grandes probabilités
    """
    
    def __init__(self):
        self.name = "margin"
    
    def select(self, model, X_unlabeled, n_instances):
        """
        Sélectionne les échantillons avec la plus petite marge entre 
        les deux classes les plus probables.
        
        Args:
            model: modèle entraîné avec méthode predict_proba
            X_unlabeled: données non étiquetées (shape: n_samples, n_features)
            n_instances: nombre d'échantillons à sélectionner
        
        Returns:
            indices: array des indices des échantillons sélectionnés
        
        Example:
            >>> strategy = MarginSamplingQueryStrategy()
            >>> indices = strategy.select(model, X_unlabeled, 10)
        """
        if n_instances <= 0:
            return np.array([], dtype=int)
        
        if n_instances > len(X_unlabeled):
            n_instances = len(X_unlabeled)
        
        # Obtenir les probabilités de prédiction
        probas = model.predict_proba(X_unlabeled)
        
        # Trier les probabilités pour chaque exemple (ordre croissant)
        probas_sorted = np.sort(probas, axis=1)
        
        # Marge = différence entre les deux meilleures probabilités
        # probas_sorted[:, -1] = plus grande probabilité
        # probas_sorted[:, -2] = deuxième plus grande probabilité
        margins = probas_sorted[:, -1] - probas_sorted[:, -2]
        
        # Sélectionner les indices avec les plus petites marges
        indices = np.argsort(margins)[:n_instances]
        
        return indices
    
    def __repr__(self):
        return f"MarginSamplingQueryStrategy(name='{self.name}')"