from optuna import Trial
from category_encoders import WOEEncoder
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
  StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from typing import List
from typing import Optional

from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import roc_auc_score, make_scorer

from pandas import DataFrame, Series
import numpy as np
from optuna import create_study
from sklearn import datasets
#from sklego.preprocessing import ColumnSelector

import logging

logger = logging.getLogger("basic")
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ch = logging.FileHandler("optuna_studies.log")
ch.setFormatter(formatter)

logger.addHandler(ch)

logger.info("Get data")
cancer = datasets.load_breast_cancer()
X_full = DataFrame(cancer['data'], columns = cancer['feature_names']) 
#y_full = DataFrame(cancer['target'], columns =['Cancer'])
y_full = cancer.target
#X_full = cancer.data

X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size = 0.30, random_state = 101) 


logger.info("Define study")
def instantiate_numerical_simple_imputer(trial : Trial, fill_value : int=-1) -> SimpleImputer:
  strategy = trial.suggest_categorical(
    'numerical_strategy', ['mean', 'median', 'most_frequent', 'constant']
  )
  return SimpleImputer(strategy=strategy, fill_value=fill_value)

def instantiate_categorical_simple_imputer(trial : Trial, fill_value : str='missing') -> SimpleImputer:
  strategy = trial.suggest_categorical(
    'categorical_strategy', ['most_frequent', 'constant']
  )
  return SimpleImputer(strategy=strategy, fill_value=fill_value)

def instantiate_woe_encoder(trial : Trial) -> WOEEncoder:
  params = {
    'sigma': trial.suggest_float('woe_sigma', 0.001, 5),
    'regularization': trial.suggest_float('woe_regularization', 0, 5),
    'randomized': trial.suggest_categorical('woe_randomized', [True, False])
  }
  return WOEEncoder(**params)


def instantiate_ordinal_encoder(trial : Trial) -> OrdinalEncoder:
  params = {
  }
  return OrdinalEncoder(**params)


def instantiate_onehot_encoder(trial : Trial) -> OneHotEncoder:
  params = {
  }
  return OneHotEncoder(**params)


Encoder = (
  OrdinalEncoder |
  OneHotEncoder |
  WOEEncoder
)


def instantiate_encoder(trial : Trial) -> Encoder:
  method = trial.suggest_categorical(
    'encoding_method', ['ordinal', 'onehot', 'woe']
  )
  
  if method=='ordinal':
    encoder = instantiate_ordinal_encoder(trial)
  elif method=='onehot':
    encoder = instantiate_onehot_encoder(trial)
  elif method=='woe':
    encoder = instantiate_woe_encoder(trial)
  
  return encoder


def instantiate_robust_scaler(trial : Trial) -> RobustScaler:
  params = {
    'with_centering': trial.suggest_categorical(
      'rb_with_centering', [True, False]
    ),
    'with_scaling': trial.suggest_categorical(
      'rb_with_scaling', [True, False]
    )
  }
  return RobustScaler(**params)


def instantiate_standard_scaler(trial : Trial) -> StandardScaler:
  params = {
    'with_mean': trial.suggest_categorical(
      'sd_with_mean', [True, False]
    ),
    'with_std': trial.suggest_categorical(
      'sd_with_std', [True, False]
    )
  }
  return StandardScaler(**params)


def instantiate_minmax_scaler(trial : Trial) -> MinMaxScaler:
  params = {
  }
  return MinMaxScaler(**params)


def instantiate_maxabs_scaler(trial : Trial) -> MaxAbsScaler:
  params = {
  }
  return MaxAbsScaler(**params)


Scaler = (
  StandardScaler |
  MinMaxScaler |
  MaxAbsScaler |
  RobustScaler
)


def instantiate_scaler(trial : Trial) -> Scaler:
  method = trial.suggest_categorical(
    'scaling_method', ['standard', 'minmax', 'maxabs', 'robust']
  )
  
  if method=='standard':
    scaler = instantiate_standard_scaler(trial)
  elif method=='minmax':
    scaler = instantiate_minmax_scaler(trial)
  elif method=='maxabs':
    scaler = instantiate_maxabs_scaler(trial)
  elif method=='robust':
    scaler = instantiate_robust_scaler(trial)
  
  return scaler


def choose_columns(trial : Trial, columns : list[str]) -> list[str]:
  choose = lambda column: trial.suggest_categorical(column, [True, False])
  choices = [*filter(choose, columns)]
  return choices

#def instantiate_column_selector(trial : Trial, columns : list[str]) -> ColumnSelector:
#  choose = lambda column: trial.suggest_categorical(column, [True, False])
#  choices = [*filter(choose, columns)]
#  selector = ColumnSelector(choices)
#  return selector

def instantiate_numerical_pipeline(trial : Trial) -> Pipeline:
  pipeline = Pipeline([
    ('imputer', instantiate_numerical_simple_imputer(trial)),
    ('scaler', instantiate_scaler(trial))
  ])
  return pipeline

def instantiate_categorical_pipeline(trial : Trial) -> Pipeline:
  pipeline = Pipeline([
    ('imputer', instantiate_categorical_simple_imputer(trial)),
    ('encoder', instantiate_encoder(trial))
  ])
  return pipeline


def instantiate_processor(trial : Trial, numerical_columns : List[str], categorical_columns : List[str]) -> ColumnTransformer:
  
  numerical_pipeline = instantiate_numerical_pipeline(trial)
  categorical_pipeline = instantiate_categorical_pipeline(trial)

  selected_numerical_columns = choose_columns(trial, numerical_columns)
  selected_categorical_columns = choose_columns(trial, categorical_columns)

  processor = ColumnTransformer([
    ('numerical_pipeline', numerical_pipeline, selected_numerical_columns),
    ('categorical_pipeline', categorical_pipeline, selected_categorical_columns)
  ])
  
  return processor


def instantiate_extra_forest(trial : Trial) -> ExtraTreesClassifier:
  params = {
    'n_estimators': trial.suggest_int('ef_n_estimators', 50, 1000),
    'max_depth': trial.suggest_int('ef_max_depth', 1, 20),
    'max_features': trial.suggest_float('ef_max_features', 0, 1),
    'bootstrap': trial.suggest_categorical('ef_bootstrap', [True, False]),
    'n_jobs': -1,
    'random_state': 42
  }
  return ExtraTreesClassifier(**params)


def instantiate_logistic_regression(trial : Trial) -> LogisticRegression:
  params = {
    'penalty': trial.suggest_categorical('lr_penalty', ['l1', 'l2']),
    'C': trial.suggest_float('rl_C', 0.001, 4, log=True),
    'solver': 'liblinear',
    #'n_jobs': -1,
    'random_state': 42
  }
  return LogisticRegression(**params)


def instantiate_random_forest(trial : Trial) -> RandomForestClassifier:
  params = {
    'n_estimators': trial.suggest_int('rf_n_estimators', 2, 10),
    'max_features': trial.suggest_int('rf_max_features', 2, 100),
    'n_jobs': -1,
    'random_state': 42
  }
  return RandomForestClassifier(**params)


def instantiate_svm(trial : Trial) -> SVC:
  params = {
    'C': trial.suggest_float('svc_C', 1, 1000, log=True),
    #'gamma': trial.suggest_float('svc_gamma', 0.0001, 0.001, log=True),
    'kernel': trial.suggest_categorical('svc_kernel', ['rbf','linear'])
  }
  return SVC(**params)


def instantiate_knn(trial : Trial) -> KNeighborsClassifier:
  params = {
    'n_neighbors': trial.suggest_int('knn_n_neighbors', 2, 30),
    'n_jobs': -1
  }
  return KNeighborsClassifier(**params)

Classifier = (
  RandomForestClassifier |
  ExtraTreesClassifier |
  SVC |
  LogisticRegression |
  KNeighborsClassifier
)

def instantiate_learner(trial : Trial) -> Classifier:
  algorithm = trial.suggest_categorical(
    #'algorithm', ['logistic', 'forest', 'extra_forest', 'svm', 'knn']
    'algorithm', ['logistic', 'forest', 'extra_forest', 'knn']
  )
  if algorithm=='logistic':
    model = instantiate_logistic_regression(trial)
  elif algorithm=='forest':
    model = instantiate_random_forest(trial)
  elif algorithm=='extra_forest':
    model = instantiate_extra_forest(trial)
  elif algorithm=='svm':
    model = instantiate_svm(trial)
  elif algorithm=='knn':
    model = instantiate_knn(trial)
  
  return model


def instantiate_model(trial : Trial, numerical_columns : List[str], categorical_columns : List[str]) -> Pipeline:
  
  processor = instantiate_processor(
    trial, numerical_columns, categorical_columns
  )
  learner = instantiate_learner(trial)
  
  model = Pipeline([
    ('processor', processor),
    ('model', learner)
  ])
  
  return model

def objective(trial : Trial, X : DataFrame, y : np.ndarray | Series, numerical_columns : Optional[List[str]]=None, categorical_columns : Optional[List[str]]=None, random_state : int=42) -> float:
  if numerical_columns is None:
    numerical_columns = [
      *DataFrame(X).select_dtypes(exclude=['object', 'category']).columns
    ]
  
  if categorical_columns is None:
    categorical_columns = [
      *DataFrame(X).select_dtypes(include=['object', 'category']).columns
    ]
  
  model = instantiate_model(trial, numerical_columns, categorical_columns)
  
  kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
  roc_auc_scorer = make_scorer(roc_auc_score, needs_proba=True)
  scores = cross_val_score(model, X, y, scoring=roc_auc_scorer, cv=kf)
  
  return np.min([np.mean(scores), np.median([scores])])


study = create_study(study_name='optimization', direction='maximize')

#logger.info("Type of y_train: "+str(type(y_train)))
study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=100)

logger.info(study.best_params)

best_trial = study.best_trial
logger.info(best_trial)

numerical_columns = [
  *DataFrame(X_train).select_dtypes(exclude=['object', 'category']).columns
  ]
  
categorical_columns = [
  *DataFrame(X_train).select_dtypes(include=['object', 'category']).columns
  ]

model = instantiate_model(best_trial, numerical_columns, categorical_columns)
model.fit(X_train, y_train)

probabilities = model.predict_proba(X_test)[:, 1]
score = roc_auc_score(y_test, probabilities)
logger.info("Score on test set: "+str(score))

model.fit(X_full, y_full)

