"""
Stratégie Random Sampling
Sélectionne des exemples aléatoirement (baseline)
"""

import numpy as np


class RandomSamplingQueryStrategy:
    """
    Stratégie baseline: sélection aléatoire.
    
    Cette stratégie ne prend pas en compte l'incertitude du modèle
    et sélectionne des exemples aléatoirement. Elle sert de baseline
    pour comparer les performances des autres stratégies.
    """
    
    def __init__(self):
        self.name = "random"
    
    def select(self, model, X_unlabeled, n_instances):
        """
        Sélectionne des échantillons aléatoirement.
        
        Args:
            model: modèle (non utilisé, paramètre pour compatibilité)
            X_unlabeled: données non étiquetées (shape: n_samples, n_features)
            n_instances: nombre d'échantillons à sélectionner
        
        Returns:
            indices: array des indices des échantillons sélectionnés
        
        Example:
            >>> strategy = RandomSamplingQueryStrategy()
            >>> indices = strategy.select(None, X_unlabeled, 10)
        
        Note:
            Le paramètre 'model' n'est pas utilisé mais maintenu pour
            avoir une interface cohérente avec les autres stratégies.
        """
        if n_instances <= 0:
            return np.array([], dtype=int)
        
        n_available = len(X_unlabeled)
        
        if n_instances > n_available:
            n_instances = n_available
        
        # Sélection aléatoire sans remplacement
        indices = np.random.choice(
            n_available, 
            size=n_instances, 
            replace=False
        )
        
        return indices
    
    def __repr__(self):
        return f"RandomSamplingQueryStrategy(name='{self.name}')"