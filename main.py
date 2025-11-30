import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Ajouter la racine du projet au chemin Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_arabic_dataset
from src.active_learner import ActiveLearner
from config import STRATEGIES, RESULTS_DIR


def plot_comparison(results_dict, metric="accuracy"):
    """
    Compare les courbes d'apprentissage pour différentes stratégies.
    
    Args:
        results_dict: dictionnaire {strategy_name: history}
        metric: métrique à afficher ("accuracy", "f1_macro", "f1_micro")
    """
    plt.figure(figsize=(14, 8))
    
    # Utiliser des couleurs et styles différents pour chaque stratégie
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    markers = ['o', 's', '^', 'D', 'v', '*', 'p', 'h', 'x']
    
    for idx, (strategy_name, history) in enumerate(results_dict.items()):
        plt.plot(
            history["n_labeled"], 
            history[metric], 
            marker=markers[idx % len(markers)], 
            label=strategy_name,
            linewidth=2.5,
            markersize=6,
            color=colors[idx],
            alpha=0.8
        )
    
    plt.xlabel("Nombre d'exemples étiquetés", fontsize=14, fontweight='bold')
    plt.ylabel(metric.replace("_", " ").title(), fontsize=14, fontweight='bold')
    plt.title(f"Comparaison des stratégies - {metric.replace('_', ' ').title()}", 
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Sauvegarder
    output_path = os.path.join(RESULTS_DIR, f"comparison_{metric}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Graphique sauvegardé: {output_path}")
    plt.close()


def save_results_table(results_dict):
    """Sauvegarde un tableau récapitulatif des résultats."""
    output_path = os.path.join(RESULTS_DIR, "results_summary.txt")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write(" RÉSULTATS COMPARATIFS - ACTIVE LEARNING SUR MANUSCRITS ARABES\n")
        f.write(f" Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*100 + "\n\n")
        
        # En-tête
        f.write(f"{'Stratégie':<30} {'Accuracy':>12} {'F1-Macro':>12} {'F1-Micro':>12} {'N_labeled':>12}\n")
        f.write("-"*100 + "\n")
        
        # Résultats pour chaque stratégie (triés par accuracy)
        sorted_results = sorted(
            results_dict.items(), 
            key=lambda x: x[1]['accuracy'][-1], 
            reverse=True
        )
        
        for strategy_name, history in sorted_results:
            acc = history['accuracy'][-1]
            f1_macro = history['f1_macro'][-1]
            f1_micro = history['f1_micro'][-1]
            n_labeled = history['n_labeled'][-1]
            
            f.write(f"{strategy_name:<30} {acc:>12.4f} {f1_macro:>12.4f} {f1_micro:>12.4f} {n_labeled:>12}\n")
        
        f.write("="*100 + "\n\n")
        
        # Statistiques supplémentaires
        f.write("STATISTIQUES DÉTAILLÉES\n")
        f.write("-"*100 + "\n\n")
        
        for strategy_name, history in sorted_results:
            f.write(f"Stratégie: {strategy_name}\n")
            f.write(f"  - Accuracy initiale:  {history['accuracy'][0]:.4f}\n")
            f.write(f"  - Accuracy finale:    {history['accuracy'][-1]:.4f}\n")
            f.write(f"  - Gain:              +{history['accuracy'][-1] - history['accuracy'][0]:.4f}\n")
            f.write(f"  - F1-macro final:     {history['f1_macro'][-1]:.4f}\n")
            f.write(f"  - F1-micro final:     {history['f1_micro'][-1]:.4f}\n")
            f.write("\n")
        
        f.write("="*100 + "\n")
    
    print(f"  Tableau récapitulatif sauvegardé: {output_path}")


def save_detailed_csv(results_dict):
    """Sauvegarde les résultats détaillés en CSV pour analyse ultérieure."""
    import csv
    
    output_path = os.path.join(RESULTS_DIR, "detailed_results.csv")
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # En-tête
        writer.writerow(['Strategy', 'Iteration', 'N_Labeled', 'Accuracy', 'F1_Macro', 'F1_Micro'])
        
        # Données
        for strategy_name, history in results_dict.items():
            for i in range(len(history['n_labeled'])):
                writer.writerow([
                    strategy_name,
                    i + 1,
                    history['n_labeled'][i],
                    history['accuracy'][i],
                    history['f1_macro'][i],
                    history['f1_micro'][i]
                ])
    
    print(f"  Résultats détaillés sauvegardés: {output_path}")


def main():
    print("="*100)
    print(" ACTIVE LEARNING - MANUSCRIT ARABE ")
    print(" Comparaison de multiples stratégies")
    print("="*100)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Nombre de stratégies à tester: {len(STRATEGIES)}")
    print(f"Stratégies: {', '.join(STRATEGIES)}")
    print("="*100)
    
    # Charger les données
    print("\n Chargement des données...")
    X_train, X_test, y_train, y_test, class_names = load_arabic_dataset()
    print(f" Données chargées: {len(class_names)} classes")
    print(f"  - Train: {len(X_train)} exemples")
    print(f"  - Test:  {len(X_test)} exemples")
    print(f"  - Classes: {', '.join(class_names[:5])}{'...' if len(class_names) > 5 else ''}")
    
    # Dictionnaire pour stocker les résultats
    results_dict = {}
    
    # Tester chaque stratégie
    for i, strategy in enumerate(STRATEGIES, 1):
        print(f"\n{'='*100}")
        print(f"  [{i}/{len(STRATEGIES)}] Stratégie: {strategy.upper()}")
        print(f"{'='*100}")
        
        try:
            learner = ActiveLearner(
                X_train.copy(), y_train.copy(),  # Copier pour éviter les modifications
                X_test, y_test, 
                strategy_name=strategy
            )
            history = learner.run()
            results_dict[strategy] = history
            
            # Afficher résumé
            print("\n Résumé:")
            print(f"   Accuracy finale:    {history['accuracy'][-1]:.4f}")
            print(f"   F1-macro final:     {history['f1_macro'][-1]:.4f}")
            print(f"   F1-micro final:     {history['f1_micro'][-1]:.4f}")
            print(f"  Exemples annotés:   {history['n_labeled'][-1]}")
            
        except Exception as e:
            print(f"\n ERREUR avec la stratégie '{strategy}': {e}")
            import traceback
            traceback.print_exc()
            print(f"\n  Passage à la stratégie suivante...\n")
    
    # Vérifier qu'on a au moins des résultats
    if not results_dict:
        print("\n Aucune stratégie n'a pu être exécutée avec succès!")
        return
    
    # Comparaison finale
    print("\n" + "="*100)
    print(" COMPARAISON FINALE DES STRATÉGIES")
    print("="*100)
    
    # Afficher le classement
    sorted_strategies = sorted(
        results_dict.items(), 
        key=lambda x: x[1]['accuracy'][-1], 
        reverse=True
    )
    
    print("\n Classement par Accuracy finale:")
    for rank, (strategy, history) in enumerate(sorted_strategies, 1):
        medal = "1. " if rank == 1 else "2. " if rank == 2 else "3. " if rank == 3 else f"{rank}."
        print(f"  {medal} {strategy:<30} Accuracy: {history['accuracy'][-1]:.4f}  "
              f"F1-macro: {history['f1_macro'][-1]:.4f}")
    
    # Générer les graphiques
    print("\nGénération des graphiques de comparaison...")
    try:
        plot_comparison(results_dict, metric="accuracy")
        plot_comparison(results_dict, metric="f1_macro")
        plot_comparison(results_dict, metric="f1_micro")
        print(" Graphiques générés avec succès")
    except Exception as e:
        print(f"Erreur lors de la génération des graphiques: {e}")
    
    # Sauvegarder les résultats
    print("\n Sauvegarde des résultats...")
    try:
        save_results_table(results_dict)
        save_detailed_csv(results_dict)
        print(" Résultats sauvegardés avec succès")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde: {e}")
    
    print("\n" + "="*100)
    print(f" Tous les résultats sont dans: {RESULTS_DIR}")
    print("="*100)
    print("\n Fichiers générés:")
    print(f"  - comparison_accuracy.png")
    print(f"  - comparison_f1_macro.png")
    print(f"  - comparison_f1_micro.png")
    print(f"  - results_summary.txt")
    print(f"  - detailed_results.csv")
    print("\n")


if __name__ == "__main__":
    main()