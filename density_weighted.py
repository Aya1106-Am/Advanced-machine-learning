"""
Stratégie Density Weighted Methods
Combine l'incertitude avec la densité des exemples pour éviter les outliers
"""

import numpy as np
from scipy.stats import entropy
from sklearn.metrics.pairwise import euclidean_distances


class DensityWeightedStrategy:
    """
    Density Weighted Methods: combine l'incertitude avec la représentativité.
    
    Cette stratégie pondère l'incertitude par la densité locale pour:
    1. Favoriser les exemples incertains (comme Entropy)
    2. Favoriser les exemples représentatifs (haute densité)
    3. Éviter les outliers isolés
    
    Score: informativeness * representativeness
    où:
    - informativeness = incertitude (entropie)
    - representativeness = densité locale (similarité aux autres exemples)
    """
    
    def __init__(self, beta=1.0):
        """
        Args:
            beta: coefficient de pondération de la densité
                  beta=0: pure uncertainty sampling
                  beta=1: balance égale
                  beta>1: favorise la représentativité
        """
        self.name = "density_weighted"
        self.beta = beta
    
    def select(self, model, X_unlabeled, n_instances):
        """
        Sélectionne les échantillons incertains et représentatifs.
        
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
        
        # 1. INFORMATIVENESS: Calculer l'incertitude (entropie)
        probas = model.predict_proba(X_unlabeled)
        uncertainties = entropy(probas.T)
        
        # Normaliser l'incertitude
        if np.max(uncertainties) > 0:
            uncertainties_norm = uncertainties / np.max(uncertainties)
        else:
            uncertainties_norm = uncertainties
        
        # 2. REPRESENTATIVENESS: Calculer la densité locale
        # Méthode: moyenne des similarités aux k plus proches voisins
        
        # Pour optimiser, on échantillonne si le dataset est trop grand
        if len(X_unlabeled) > 1000:
            # Échantillonner 1000 exemples pour calculer les distances
            sample_indices = np.random.choice(
                len(X_unlabeled), 
                size=min(1000, len(X_unlabeled)), 
                replace=False
            )
            X_sample = X_unlabeled[sample_indices]
        else:
            X_sample = X_unlabeled
        
        # Calculer les distances euclidiennes
        distances = euclidean_distances(X_unlabeled, X_sample)
        
        # Pour chaque point, calculer la densité comme l'inverse de la distance moyenne
        # aux k plus proches voisins
        k = min(10, len(X_sample))  # 10 plus proches voisins
        
        # Trier les distances et prendre les k plus proches
        sorted_distances = np.sort(distances, axis=1)
        k_nearest_distances = sorted_distances[:, 1:k+1]  # Exclure le point lui-même (distance=0)
        
        # Densité = inverse de la distance moyenne
        avg_distances = np.mean(k_nearest_distances, axis=1)
        # Éviter la division par zéro
        densities = 1.0 / (avg_distances + 1e-6)
        
        # Normaliser la densité
        if np.max(densities) > 0:
            densities_norm = densities / np.max(densities)
        else:
            densities_norm = densities
        
        # 3. COMBINER: Score final
        # Score = uncertainty^(1) * density^(beta)
        combined_scores = uncertainties_norm * (densities_norm ** self.beta)
        
        # Sélectionner les indices avec les plus grands scores
        indices = np.argsort(-combined_scores)[:n_instances]
        
        return indices
    
    def __repr__(self):
        return f"DensityWeightedStrategy(name='{self.name}', beta={self.beta})"