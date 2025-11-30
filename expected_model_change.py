"""
Stratégie Expected Model Change (EMC)
Sélectionne les exemples qui causeraient le plus grand changement dans le modèle
"""

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class ExpectedModelChangeStrategy:
    """
    Expected Model Change: sélectionne les exemples qui changeraient le plus le modèle.
    
    Cette stratégie estime l'impact qu'aurait l'ajout d'un exemple sur les paramètres
    du modèle. Pour les SVM, on utilise une approximation basée sur la distance
    aux vecteurs de support et l'incertitude de prédiction.
    
    Formule approximative: EMC(x) = uncertainty(x) * gradient_magnitude(x)
    """
    
    def __init__(self):
        self.name = "expected_model_change"
    
    def select(self, model, X_unlabeled, n_instances):
        """
        Sélectionne les échantillons qui changeraient le plus le modèle.
        
        Args:
            model: modèle entraîné (SVM ou autre)
            X_unlabeled: données non étiquetées
            n_instances: nombre d'échantillons à sélectionner
        
        Returns:
            indices: array des indices sélectionnés
        """
        if n_instances <= 0:
            return np.array([], dtype=int)
        
        if n_instances > len(X_unlabeled):
            n_instances = len(X_unlabeled)
        
        # Obtenir les probabilités
        probas = model.predict_proba(X_unlabeled)
        
        # Incertitude (entropie)
        from scipy.stats import entropy
        uncertainties = entropy(probas.T)
        
        # Approximation du gradient basée sur l'incertitude et la confiance
        # Plus la prédiction est incertaine, plus l'impact potentiel est grand
        max_probs = np.max(probas, axis=1)
        second_max_probs = np.partition(probas, -2, axis=1)[:, -2]
        
        # Marge entre les deux meilleures prédictions
        margins = max_probs - second_max_probs
        
        # Score EMC: combine incertitude et petite marge
        # (petite marge = grand changement potentiel)
        emc_scores = uncertainties * (1 - margins)
        
        # Pour SVM, on peut aussi utiliser la distance aux vecteurs de support
        if hasattr(model, 'support_vectors_'):
            support_vectors = model.support_vectors_
            # Distance minimale à chaque vecteur de support
            distances = euclidean_distances(X_unlabeled, support_vectors)
            min_distances = np.min(distances, axis=1)
            # Normaliser les distances
            if np.max(min_distances) > 0:
                min_distances = min_distances / np.max(min_distances)
            # Combiner avec le score EMC (plus proche = plus d'impact)
            emc_scores = emc_scores * (1 + (1 - min_distances))
        
        # Sélectionner les indices avec les plus grands scores EMC
        indices = np.argsort(-emc_scores)[:n_instances]
        
        return indices
    
    def __repr__(self):
        return f"ExpectedModelChangeStrategy(name='{self.name}')"