"""
Stratégie Variance Reduction
Sélectionne les exemples qui réduiraient le plus la variance du modèle
"""

import numpy as np
from scipy.stats import entropy


class VarianceReductionStrategy:
    """
    Variance Reduction: sélectionne les exemples qui minimiseraient
    la variance des prédictions du modèle.
    
    Cette stratégie cherche à réduire l'incertitude globale du modèle en
    sélectionnant les exemples dans les régions où les prédictions sont
    les plus variables.
    
    Approximation: on combine l'entropie locale avec la dispersion des probabilités
    """
    
    def __init__(self):
        self.name = "variance_reduction"
    
    def select(self, model, X_unlabeled, n_instances):
        """
        Sélectionne les échantillons qui réduiraient le plus la variance.
        
        Args:
            model: modèle entraîné avec predict_proba
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
        n_samples, n_classes = probas.shape
        
        # 1. Incertitude (entropie)
        uncertainties = entropy(probas.T)
        
        # 2. Variance des probabilités prédites
        # Plus la distribution est uniforme, plus la variance est élevée
        prob_variance = np.var(probas, axis=1)
        
        # 3. Distance à la distribution uniforme
        uniform_dist = np.ones(n_classes) / n_classes
        kl_divergence = np.sum(probas * np.log(probas / uniform_dist + 1e-10), axis=1)
        
        # 4. Score de variance: combine plusieurs mesures
        # Normaliser chaque composante
        if np.max(uncertainties) > 0:
            uncertainties_norm = uncertainties / np.max(uncertainties)
        else:
            uncertainties_norm = uncertainties
            
        if np.max(prob_variance) > 0:
            prob_variance_norm = prob_variance / np.max(prob_variance)
        else:
            prob_variance_norm = prob_variance
            
        if np.max(np.abs(kl_divergence)) > 0:
            kl_divergence_norm = np.abs(kl_divergence) / np.max(np.abs(kl_divergence))
        else:
            kl_divergence_norm = np.abs(kl_divergence)
        
        # Combiner les scores (moyenne pondérée)
        variance_scores = (
            0.5 * uncertainties_norm +
            0.3 * prob_variance_norm +
            0.2 * (1 - kl_divergence_norm)  # Favoriser les distributions proches de l'uniforme
        )
        
        # Sélectionner les indices avec les plus grands scores
        indices = np.argsort(-variance_scores)[:n_instances]
        
        return indices
    
    def __repr__(self):
        return f"VarianceReductionStrategy(name='{self.name}')"
    