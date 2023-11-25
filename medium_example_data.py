import numpy as np
from pandas import DataFrame

from optuna import Trial
from optuna import TrialPruned
from optuna import create_study
from optuna.samplers import RandomSampler
from optuna.pruners import SuccessiveHalvingPruner

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn import datasets


cancer = datasets.load_breast_cancer()
X_full = DataFrame(cancer['data'], columns = cancer['feature_names']) 
#y_full = DataFrame(cancer['target'], columns =['Cancer'])
y_full = cancer.target

#log_int = lambda x, base: np.floor(np.log(x)/np.log(base)).astype(int)

def log_int(x, base : int=2):
  return np.floor(np.log(x)/np.log(base)).astype(int)

def generate_sample_numbers(y : DataFrame, base : int, n_rungs : int) -> list[int]:
  
  data_size = len(y)
  data_scale = log_int(data_size, base)
  min_scale = data_scale - n_rungs
  min_samples = base**min_scale
  
  return [
      *map(lambda scale: base**scale, range(min_scale, data_scale+1))
  ]

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


def objective(trial: Trial, X : DataFrame, y : DataFrame, seed : int=42, base : int=2, n_rungs=4) -> float:
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, shuffle=True, random_state=seed
  )
  model = instantiate_extra_trees(trial, warm_start=False)
  
  n_samples_list = generate_sample_numbers(y_train, base, n_rungs)
      
  for n_samples in n_samples_list:
    X_train_sample = X_train.sample(n_samples, random_state=seed)
    y_train_sample = DataFrame(y_train).sample(n_samples, random_state=seed)

    model.fit(X_train_sample, y_train_sample.values.ravel())

    score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    trial.report(score, n_samples)
    
    if trial.should_prune():
      raise TrialPruned()

  kfold = KFold(shuffle=True, random_state=seed)
  roc_auc = make_scorer(roc_auc_score, needs_proba=True)
  scores = cross_val_score(model, X, y, cv=kfold, scoring=roc_auc)
  return np.min([np.mean(scores), np.median(scores)])


factor = 2
study = create_study(
  direction="maximize",
  pruner=SuccessiveHalvingPruner(reduction_factor=factor),
  sampler=RandomSampler(seed=42) #not necessary, helps with reproducibility
)
study.optimize(
  lambda trial: objective(trial, X_full, y_full, base=factor, n_rungs=4), n_trials=60
)