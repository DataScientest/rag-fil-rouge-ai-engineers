# Démos projetées en masterclass

Deux formats cohabitent ici :

| Format | Fichiers | Lancement |
|---|---|---|
| Jupyter (`.ipynb`) | `demo_02` à `demo_13` | `make notebook`, puis ouvrir `demo/` |
| marimo (`.py`) | `demo_02b_attention_position.py` | `make marimo` — voir ci-dessous |

Les `.ipynb` sont enregistrés avec leurs sorties (ils se projettent sans rien exécuter). Les carnets marimo sont des fichiers Python ordinaires, avec un en-tête de dépendances.

## Lancer un carnet marimo

### Prérequis (une fois)

`uv` seulement :

```bash
uv --version
```

La cible `make marimo` passe par `uvx`, qui installe marimo à la volée dans un environnement isolé : pas besoin de l'installer globalement. Si tu préfères avoir la commande `marimo` disponible partout : `uv tool install marimo`.

### Lancement

```bash
make marimo                                # le carnet marimo par défaut
make marimo MN=notebooks/demo/autre.py     # un autre carnet
```

Les chemins passés en `MN=` sont relatifs à la racine du dépôt.

La cible équivaut à :

```bash
uvx marimo edit --sandbox notebooks/demo/demo_02b_attention_position.py
```

marimo affiche une URL (port 2718 par défaut) et ouvre le navigateur. `Ctrl+C` dans le terminal arrête le serveur. Pour changer de port : `--port 2719`.

### Pourquoi `--sandbox`

Le carnet déclare ses dépendances dans son **en-tête PEP 723** :

```python
# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo", "matplotlib", "numpy", "plotly", "torch"]
# ///
```

`--sandbox` demande à uv de construire un environnement isolé à partir de cette liste. Le `.venv` du projet n'est ni lu ni modifié, et `make install-demo` n'est pas nécessaire.

Compter quelques minutes au **premier** lancement : `torch` et ses bibliothèques CUDA pèsent 2 à 3 Go. uv les met en cache, les lancements suivants sont immédiats.

### Autres usages

```bash
# Projeter sans le code (mode application, lecture seule)
uvx marimo run --sandbox notebooks/demo/demo_02b_attention_position.py

# Serveur seul, sans ouvrir de navigateur (SSH, volet d'un éditeur)
uvx marimo edit --sandbox --headless notebooks/demo/demo_02b_attention_position.py

# Exporter une page HTML statique, à partager
uvx marimo export html --sandbox notebooks/demo/demo_02b_attention_position.py -o /tmp/demo.html
```

`marimo run` masque le code : c'est le mode à utiliser en projection.

## Ce qu'il faut savoir sur marimo

- Les cellules ne s'exécutent **pas dans l'ordre** : elles forment un graphe de dépendances. Bouger un curseur ré-exécute automatiquement les cellules qui en dépendent — c'est ce qui fait « bouger » les animations.
- **Pas d'état caché** : le carnet se ré-exécute entièrement et redonne le même résultat, dans n'importe quel ordre de clic.
- La **sauvegarde est automatique** (le fichier est réécrit à chaque modification). Il crée un dossier `__marimo__/` (cache de session), déjà ignoré par Git.
- Le fichier `.py` est la source de vérité : il se lit et se relit en diff comme du Python classique.

## Spécificités de `demo_02b_attention_position.py`

- **GPU facultatif** : un interrupteur « Calcul sur GPU (CUDA via torch) » en tête du carnet fait tourner le calcul des tenseurs sur la carte graphique. Sans CUDA, il retombe sur le CPU tout seul.
  - Avec le GPU actif, `torch` charge ~0,6 Go de RAM (contexte CUDA compris). Sur ce carnet, le GPU coûte donc de la mémoire plutôt que d'en libérer : le vrai gain vient du rendu.
  - Les animations sont en **Plotly/WebGL** pilotées par curseur (aucune image de frame embarquée), les cartes denses restent en raster.
- **Dépendances** : `torch` et `plotly` ne sont utilisés que par ce carnet ; ils ne sont pas dans le `.venv` du projet (d'où `--sandbox`).
