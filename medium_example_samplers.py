from optuna import create_study
from optuna import Trial
from cmaes import CMA
from optuna.samplers import RandomSampler, QMCSampler, TPESampler, CmaEsSampler
from optuna.visualization.matplotlib import plot_contour
from optuna.visualization import plot_contour

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import numpy as np
from pandas import DataFrame
from pandas import Series
from typing import Optional, List

cancer = datasets.load_breast_cancer()
X_full = DataFrame(cancer['data'], columns = cancer['feature_names']) 
#y_full = DataFrame(cancer['target'], columns =['Cancer'])
y_full = cancer.target


def instantiate_extra_trees(trial : Trial) -> ExtraTreesClassifier:
    params = {
        'max_features': trial.suggest_float('max_features', 1e-5, 1),
        'min_samples_leaf': trial.suggest_float('min_samples_leaf', 1e-5, 1),
        'n_jobs': -1,
        'random_state': 42
    }
    return ExtraTreesClassifier(**params)

def objective(trial : Trial, X : DataFrame, y : np.ndarray | Series, numerical_columns : Optional[List[str]]=None, categorical_columns : Optional[List[str]]=None, random_state : int=42) -> float:
  
  model = instantiate_extra_trees(trial)
  
  kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
  roc_auc_scorer = make_scorer(roc_auc_score, needs_proba=True)
  scores = cross_val_score(model, X, y, scoring=roc_auc_scorer, cv=kf)
  
  return np.min([np.mean(scores), np.median([scores])])


random_study = create_study(
    direction="maximize",
    sampler=RandomSampler(seed=42)
)
random_study.optimize(lambda trial: objective(trial, X_full, y_full), n_trials=100)

plot_contour(random_study, ['max_features', 'min_samples_leaf']).write_html("random_study.html")

qmc_study = create_study(
    direction="maximize",
    sampler=QMCSampler(seed=42),
)
qmc_study.optimize(lambda trial: objective(trial, X_full, y_full), n_trials=100)

plot_contour(qmc_study, ['max_features', 'min_samples_leaf']).write_html("qmc_study.html")

cma_study = create_study(
    direction="maximize",
    sampler=CmaEsSampler(seed=42),
)
cma_study.optimize(lambda trial: objective(trial, X_full, y_full), n_trials=100)

plot_contour(cma_study, ['max_features', 'min_samples_leaf']).write_html("cma_study.html")
tpe_study = create_study(
    direction="maximize",
    sampler=TPESampler(seed=42),
)
tpe_study.optimize(lambda trial: objective(trial, X_full, y_full), n_trials=100)

plot_contour(tpe_study, ['max_features', 'min_samples_leaf']).write_html("tpe_study.html")