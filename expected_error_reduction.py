"""
Stratégie Expected Error Reduction (EER)
Sélectionne les exemples qui réduiraient le plus l'erreur attendue
"""

import numpy as np
from scipy.stats import entropy


class ExpectedErrorReductionStrategy:
    """
    Expected Error Reduction: sélectionne les exemples qui minimiseraient
    l'erreur future du modèle.
    
    Cette stratégie simule l'ajout de chaque exemple avec chaque label possible,
    et estime la réduction d'erreur résultante. Version simplifiée pour 
    des raisons de performance.
    
    Approximation: EER(x) ≈ entropy(p(y|x)) * avg_confidence
    """
    
    def __init__(self):
        self.name = "expected_error_reduction"
    
    def select(self, model, X_unlabeled, n_instances):
        """
        Sélectionne les échantillons qui réduiraient le plus l'erreur.
        
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
        
        # Calculer l'entropie (incertitude du modèle)
        uncertainties = entropy(probas.T)
        
        # Pour chaque exemple, estimer la réduction d'erreur
        # Version simplifiée: on utilise l'incertitude pondérée par 
        # la distribution des probabilités
        eer_scores = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Pour chaque classe possible
            expected_error_reduction = 0
            for c in range(n_classes):
                # Probabilité que l'exemple appartienne à la classe c
                p_c = probas[i, c]
                
                # Si on ajoute cet exemple avec le label c,
                # la réduction d'erreur serait proportionnelle à:
                # - l'incertitude actuelle
                # - la probabilité de cette classe
                # - l'impact sur les autres prédictions
                
                # Approximation: réduction attendue si c'est la vraie classe
                expected_error_reduction += p_c * uncertainties[i] * (1 - p_c)
            
            eer_scores[i] = expected_error_reduction
        
        # Sélectionner les indices avec les plus grands scores EER
        indices = np.argsort(-eer_scores)[:n_instances]
        
        return indices
    
    def __repr__(self):
        return f"ExpectedErrorReductionStrategy(name='{self.name}')"