from typing import Optional
from pandas import DataFrame
import numpy as np
from optuna import Trial
from optuna import TrialPruned
from optuna import create_study
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import RandomSampler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import datasets

cancer = datasets.load_breast_cancer()
X_full = DataFrame(cancer['data'], columns = cancer['feature_names']) 
#y_full = DataFrame(cancer['target'], columns =['Cancer'])
y_full = cancer.target


def instantiate_sgd_classifier(trial : Trial) -> SGDClassifier:
  params = {
    'loss': trial.suggest_categorical('sgd_loss', ['modified_huber', 'log_loss']),
    'n_jobs': -1,
    'random_state': 42
  }
  return SGDClassifier(**params)


def objective(trial : Trial, X : DataFrame, y : np.ndarray, seed : int=42) -> Optional[float]:
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=True, random_state=seed
  )
  
  sgd = instantiate_sgd_classifier(trial)
  n_train_iter = 128
  
  for epoch in range(n_train_iter):
    sgd.partial_fit(X_train, y_train, np.unique(y_full))

    #epoch_score = roc_auc_score(y_test, sgd.predict_proba(X_test))
    epoch_score = roc_auc_score(y_test, sgd.predict_proba(X_test)[:, 1])
    trial.report(epoch_score, epoch)

    if trial.should_prune():
      raise TrialPruned()
  
  return epoch_score

study = create_study(
  direction="maximize",
  pruner=SuccessiveHalvingPruner(reduction_factor=2),
  sampler=RandomSampler(seed=42) #not necessary, helps with reproducibility
)
study.optimize(lambda trial: objective(trial, X_full, y_full), n_trials=60)

