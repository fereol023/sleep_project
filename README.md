# _sleeproj

## Project's overview

### Titre du projet : Modélisation de la durée de sommeil : Une approche de data science

1. Introduction :
Le projet vise à analyser la relation entre la durée de sommeil et les habitudes quotidiennes en utilisant des techniques de data science. 

Nous utiliserons des données provenant de l'API Kaggle pour comprendre les facteurs qui influencent la durée du sommeil, 
afin d'optimiser le bien-être et la productivité.

2. Objectifs du projet :
- Collecter des données sur la durée de sommeil et les habitudes quotidiennes des software engineers à partir de l'API Kaggle.
- Identifier les habitudes quotidiennes spécifiques à la profession de software engineer qui peuvent être liées à la durée du sommeil.
- Utiliser des techniques de data science pour analyser les données collectées et détecter des schémas ou des corrélations significatives.
- Entraîner et évaluer des modèles de machine learning spécifiques pour prédire la durée du sommeil des software engineers en fonction de leurs habitudes quotidiennes.

3. Méthodologie :

ENVIRONNEMENT VIRTUEL MYSLEEPENV -----------------------------------------
- dans gitbash : conda create -n MYSLEEPENV.yml
- ensuite : conda activate MYSLEEPENV
- ensuite : installer ipykernel si utilisation de jupyter (dans mon cas)
- ensuite selectionner le virtual env avec python 
---------------------------------------------------------------------------

a) Collecte des données :
   - source kaggle : https://www.kaggle.com/code/wilmerarltstrmberg/sleep-disorder-feature-analysis/log
   - Utiliser l'API Kaggle pour extraire des ensembles de données pertinents sur la durée de sommeil et les habitudes quotidiennes des software engineers.
   - Vérifier la qualité des données et effectuer un prétraitement initial pour éliminer les valeurs aberrantes et les données manquantes.

b) Prétraitement des données :
   - Effectuer une analyse exploratoire des données pour comprendre leur structure, identifier les variables pertinentes et détecter les éventuelles anomalies.
   - Nettoyer et normaliser les données pour garantir leur cohérence et leur qualité.

c) Analyse des données :
   - Utiliser des techniques statistiques pour analyser les corrélations entre la durée du sommeil et les habitudes quotidiennes spécifiques à la profession de software engineer, telles que les heures de travail, les heures de loisirs, l'utilisation des écrans, etc.
   - Appliquer des techniques de visualisation de données pour représenter graphiquement les résultats de l'analyse.

d) Modélisation prédictive :
   - Diviser les données en ensembles d'apprentissage et de test.
   - Entraîner plusieurs modèles de machine learning adaptés aux spécificités des software engineers, tels que :
     - Régression linéaire ou polynomiale pour estimer la durée du sommeil en fonction des habitudes quotidiennes.
     - Méthodes d'apprentissage ensembliste comme les forêts aléatoires ou les méthodes de gradient boosting pour détecter des interactions complexes entre les variables.
   - Évaluer les performances de chaque modèle en utilisant des mesures telles que l'erreur quadratique moyenne (RMSE) ou la précision des prédictions.

4. Résultats attendus :
- Identification des habitudes quotidiennes qui ont un impact significatif sur la durée du sommeil.
- Évaluation comparative des performances des modèles de machine learning dans la prédiction de la durée du sommeil des software engineers.
- Estimation de la durée de sommeil en fonction de leurs habitudes quotidiennes.



## Overview for Kedro...~~~~~~~~~~~~~~~~~~ 

This is your new Kedro project, which was generated using `Kedro 0.18.10`.

Take a look at the [Kedro documentation](https://docs.kedro.org) to get started.

## Rules and guidelines

In order to get the best out of the template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a data engineering convention
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

## How to install dependencies

Declare any dependencies in `src/requirements.txt` for `pip` installation and `src/environment.yml` for `conda` installation.

To install them, run:

```
pip install -r src/requirements.txt
```

## How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

## How to test your Kedro project

Have a look at the file `src/tests/test_run.py` for instructions on how to write your tests. You can run your tests as follows:

```
kedro test
```

To configure the coverage threshold, go to the `.coveragerc` file.

## Project dependencies

To generate or update the dependency requirements for your project:

```
kedro build-reqs
```

This will `pip-compile` the contents of `src/requirements.txt` into a new file `src/requirements.lock`. You can see the output of the resolution by opening `src/requirements.lock`.

After this, if you'd like to update your project requirements, please update `src/requirements.txt` and re-run `kedro build-reqs`.

[Further information about project dependencies](https://docs.kedro.org/en/stable/kedro_project_setup/dependencies.html#project-specific-dependencies)

## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `context`, `catalog`, and `startup_error`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so once you have run `pip install -r src/requirements.txt` you will not need to take any extra steps before you use them.

### Jupyter
To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

### JupyterLab
To use JupyterLab, you need to install it:

```
pip install jupyterlab
```

You can also start JupyterLab:

```
kedro jupyter lab
```

### IPython
And if you want to run an IPython session:

```
kedro ipython
```

### How to convert notebook cells to nodes in a Kedro project
You can move notebook code over into a Kedro project structure using a mixture of [cell tagging](https://jupyter-notebook.readthedocs.io/en/stable/changelog.html#release-5-0-0) and Kedro CLI commands.

By adding the `node` tag to a cell and running the command below, the cell's source code will be copied over to a Python file within `src/<package_name>/nodes/`:

```
kedro jupyter convert <filepath_to_my_notebook>
```
> *Note:* The name of the Python file matches the name of the original notebook.

Alternatively, you may want to transform all your notebooks in one go. Run the following command to convert all notebook files found in the project root directory and under any of its sub-folders:

```
kedro jupyter convert --all
```

### How to ignore notebook output cells in `git`
To automatically strip out all output cell contents before committing to `git`, you can run `kedro activate-nbstripout`. This will add a hook in `.git/config` which will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be retained locally.

## Package your Kedro project

[Further information about building project documentation and packaging your project](https://docs.kedro.org/en/stable/tutorial/package_a_project.html)
