# 🔍 Simulateur de Diagnostic Optique (Combinatorial Group Testing)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Dash](https://img.shields.io/badge/Dash-Framework-008080)
![Plotly](https://img.shields.io/badge/Plotly-Graphing-3f4f75)
![NetworkX](https://img.shields.io/badge/NetworkX-Graph_Theory-lightgrey)

Application web interactive permettant de simuler, visualiser et analyser le diagnostic de pannes (liens coupés) dans les réseaux tout-optiques. Ce projet implémente et compare des approches de diagnostic naïves et des stratégies avancées basées sur le **Combinatorial Group Testing (CGT)** avec une complexité de sondage logarithmique.

## 📖 À propos du projet

Dans les réseaux optiques, tester chaque câble un par un prend un temps linéaire ($O(N)$). Ce simulateur démontre comment le *Combinatorial Group Testing* (ou procédure LTP) permet d'envoyer des sondes (lasers) qui traversent plusieurs arêtes simultanément pour identifier une panne unique avec un minimum de tests, s'approchant d'une complexité de **$O(\log N)$**.


## ✨ Fonctionnalités Principales

* **4 Topologies de Réseau supportées :**
  * **Linéaire :** Algorithme de la fenêtre glissante.
  * **Grille (n×n) :** Algorithme hiérarchique (Validation des axes, sondages globaux, sondages par échelles/indices k).
  * **Complet (Hub & Spoke) :** Sondages de hub central, puis sondages les arêtes qui restent depuis ce hub sain.
  * **Arbre (Tree) :** Implémentation  utilise la **Heavy-Light Decomposition (HLD)** pour redresser le graphe en "Chemins Préférés".
* **Comparaison des Stratégies :** Basculez en un clic entre une recherche unitaire (Naïve) et une recherche combinatoire (LTP/CGT).
* **Moteur d'Animation Fluide :** Visualisez le trajet exact de la sonde grâce à une interpolation linéaire (mouvement continu du laser de nœud en nœud) et observez le blocage en temps réel lors du crash sur une arête défaillante.
* **Rapport de Diagnostic Mathématique :** Génération automatique d'un rapport détaillant les matrices binaires, les syndromes calculés, et la déduction spatiale par théorie des ensembles (Innocentation et Règle du goulot d'étranglement).

## 🛠️ Prérequis et Installation

Assurez-vous d'avoir [Python 3.8+](https://www.python.org/downloads/) installé sur votre machine.

1. **Clonez le dépôt ou téléchargez les fichiers :**
   ```bash
   git clone https://github.com/oussamahakik/Detection-de-panne-approche-non-adaptative-.git
   cd simulateur-diagnostic-optique
