from optuna import Trial
from optuna import TrialPruned
from optuna import create_study
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import RandomSampler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn import datasets
from pandas import DataFrame
import numpy as np
from typing import Optional

cancer = datasets.load_breast_cancer()
X_full = DataFrame(cancer['data'], columns = cancer['feature_names']) 
#y_full = DataFrame(cancer['target'], columns =['Cancer'])
y_full = cancer.target


def instantiate_extra_trees(trial : Trial, warm_start : bool=False) -> ExtraTreesClassifier:
  params = {
    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
    'max_depth': trial.suggest_int('max_depth', 1, 20),
    'max_features': trial.suggest_float('max_features', 0, 1),
    'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
    'warm_start': warm_start,
    'n_jobs': -1,
    'random_state': 42
  }
  return ExtraTreesClassifier(**params)


def objective(trial : Trial, X : DataFrame, y : DataFrame, seed : int=42) -> Optional[float]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=True, random_state=seed
    )
    
    model = instantiate_extra_trees(trial, warm_start=True)
    n_estimators = model.get_params().get('n_estimators')
    min_estimators = 100
    
    for num_estimators in range(min_estimators, n_estimators + 1):
        model.set_params(n_estimators=num_estimators)
        model.fit(X_train, y_train)
        
        score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        trial.report(score, num_estimators)
    
        if trial.should_prune():
            raise TrialPruned()

    kfold = KFold(shuffle=True, random_state=seed)
    roc_auc = make_scorer(roc_auc_score, needs_proba=True)
    scores = cross_val_score(model, X, y, cv=kfold, scoring=roc_auc)
    
    return np.min([np.mean(scores), np.median(scores)])

study = create_study(
  direction="maximize",
  pruner=SuccessiveHalvingPruner(reduction_factor=2),
  sampler=RandomSampler(seed=42) #not necessary, helps with reproducibility
)
study.optimize(lambda trial: objective(trial, X_full, y_full), n_trials=60)
