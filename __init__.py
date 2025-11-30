"""
Module de stratégies de requête pour l'Active Learning
"""

from .entropy_strategy import EntropySamplingQueryStrategy
from .qbc_strategy import QBCQueryStrategy
from .least_confidence_strategy import LeastConfidenceQueryStrategy
from .margin_strategy import MarginSamplingQueryStrategy
from .random_strategy import RandomSamplingQueryStrategy
from .expected_model_change import ExpectedModelChangeStrategy
from .expected_error_reduction import ExpectedErrorReductionStrategy
from .variance_reduction import VarianceReductionStrategy
from .density_weighted import DensityWeightedStrategy


def get_query_strategy(strategy_name: str):
    """
    Factory pour obtenir une stratégie de requête par son nom.
    
    Args:
        strategy_name: nom de la stratégie 
                      ("entropy", "QBC", "least_confidence", "margin", "random",
                       "expected_model_change", "expected_error_reduction",
                       "variance_reduction", "density_weighted")
    
    Returns:
        instance de la stratégie demandée
    
    Raises:
        ValueError: si la stratégie n'est pas supportée
    """
    strategies = {
        # Stratégies basiques
        "entropy": EntropySamplingQueryStrategy,
        "QBC": QBCQueryStrategy,
        "least_confidence": LeastConfidenceQueryStrategy,
        "margin": MarginSamplingQueryStrategy,
        "random": RandomSamplingQueryStrategy,
        
        # Stratégies avancées
        "expected_model_change": ExpectedModelChangeStrategy,
        "expected_error_reduction": ExpectedErrorReductionStrategy,
        "variance_reduction": VarianceReductionStrategy,
        "density_weighted": DensityWeightedStrategy,
    }
    
    if strategy_name not in strategies:
        raise ValueError(
            f"Stratégie '{strategy_name}' non supportée. "
            f"Stratégies disponibles: {list(strategies.keys())}"
        )
    
    return strategies[strategy_name]()


__all__ = [
    'get_query_strategy',
    'EntropySamplingQueryStrategy',
    'QBCQueryStrategy',
    'LeastConfidenceQueryStrategy',
    'MarginSamplingQueryStrategy',
    'RandomSamplingQueryStrategy',
    'ExpectedModelChangeStrategy',
    'ExpectedErrorReductionStrategy',
    'VarianceReductionStrategy',
    'DensityWeightedStrategy',
]