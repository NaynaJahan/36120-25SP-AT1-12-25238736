# 36120-25SP-AT1-12-25238736

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

The NBA draft is an annual event in which teams select players from their American colleges as well as international professional leagues to join their rosters. Sport commentators and fans are very excited to follow the careers of college players and guess who will be drafted by an NBA team.

In this project, a model is built that predicts if a college basketball player will be drafted to join the NBA league based on his statistics for the current season. Best predictions from the trained ML models are submitted on a UTS-provided Kaggle Competition as well. 

The metric used to assess model performance is AUROC (Area Under ROC).

The best models artefacts can be found in the `models/` folder.
Provide the pyproject.toml  and requirements.txt files at the root of your repository.
Some functionalities are imported and used from a custom Python package that is uploaded in Testpypi. (Codes for the custom Python package can be found here: https://github.com/NaynaJahan/amla_at1_python_pkg).

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         36120-25SP-AT1-12-25238736 and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── 36120-25SP-AT1-12-25238736   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes 36120-25SP-AT1-12-25238736 a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

