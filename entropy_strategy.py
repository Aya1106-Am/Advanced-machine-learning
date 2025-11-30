"""
Stratégie Entropy Sampling
Sélectionne les exemples avec l'entropie maximale
"""

import numpy as np
from scipy.stats import entropy


class EntropySamplingQueryStrategy:
    """
    Stratégie basée sur l'entropie maximale.
    
    L'entropie mesure l'incertitude du modèle. Plus l'entropie est élevée,
    plus le modèle est incertain sur sa prédiction.
    
    Formule: H(p) = -Σ p_i * log(p_i)
    """
    
    def __init__(self):
        self.name = "entropy"
    
    def select(self, model, X_unlabeled, n_instances):
        """
        Sélectionne les n_instances exemples avec l'entropie maximale.
        
        Args:
            model: modèle entraîné avec méthode predict_proba
            X_unlabeled: données non étiquetées (shape: n_samples, n_features)
            n_instances: nombre d'échantillons à sélectionner
        
        Returns:
            indices: array des indices des échantillons sélectionnés
        
        Example:
            >>> strategy = EntropySamplingQueryStrategy()
            >>> indices = strategy.select(model, X_unlabeled, 10)
        """
        if n_instances <= 0:
            return np.array([], dtype=int)
        
        if n_instances > len(X_unlabeled):
            n_instances = len(X_unlabeled)
        
        # Obtenir les probabilités de prédiction
        probas = model.predict_proba(X_unlabeled)
        
        # Calculer l'entropie pour chaque exemple
        # entropy() de scipy calcule l'entropie sur les lignes par défaut
        entropies = entropy(probas.T)
        
        # Sélectionner les indices avec les plus grandes entropies
        indices = np.argsort(entropies)[-n_instances:]
        
        return indices
    
    def __repr__(self):
        return f"EntropySamplingQueryStrategy(name='{self.name}')"