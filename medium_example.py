from optuna import Trial
from category_encoders import WOEEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import ExtraTreesClassifier
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

(X_full, y_full) = datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X_full,y_full,test_size=0.25)

print(DataFrame(X_full).head())
print(DataFrame(X_full).describe())

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
    'sigma': trial.suggest_float('sigma', 0.001, 5),
    'regularization': trial.suggest_float('regularization', 0, 5),
    'randomized': trial.suggest_categorical('randomized', [True, False])
  }
  return WOEEncoder(**params)


def instantiate_robust_scaler(trial : Trial) -> RobustScaler:
  params = {
    'with_centering': trial.suggest_categorical(
      'with_centering', [True, False]
    ),
    'with_scaling': trial.suggest_categorical(
      'with_scaling', [True, False]
    )
  }
  return RobustScaler(**params)

#def instantiate_column_selector(trial : Trial, columns : list[str]) -> ColumnSelector:
#  choose = lambda column: trial.suggest_categorical(column, [True, False])
#  choices = [*filter(choose, columns)]
#  selector = ColumnSelector(choices)
#  return selector

def instantiate_extra_trees(trial : Trial) -> ExtraTreesClassifier:
  params = {
    'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
    'max_depth': trial.suggest_int('max_depth', 1, 20),
    'max_features': trial.suggest_float('max_features', 0, 1),
    'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
    'n_jobs': -1,
    'random_state': 42
  }
  return ExtraTreesClassifier(**params)

def instantiate_numerical_pipeline(trial : Trial) -> Pipeline:
  pipeline = Pipeline([
    ('imputer', instantiate_numerical_simple_imputer(trial)),
    ('scaler', instantiate_robust_scaler(trial))
  ])
  return pipeline

def instantiate_categorical_pipeline(trial : Trial) -> Pipeline:
  pipeline = Pipeline([
    ('imputer', instantiate_categorical_simple_imputer(trial)),
    ('encoder', instantiate_woe_encoder(trial))
  ])
  return pipeline


def instantiate_processor(trial : Trial, numerical_columns : List[str], categorical_columns : List[str]) -> ColumnTransformer:
  
  numerical_pipeline = instantiate_numerical_pipeline(trial)
  categorical_pipeline = instantiate_categorical_pipeline(trial)
  
  processor = ColumnTransformer([
    ('numerical_pipeline', numerical_pipeline, numerical_columns),
    ('categorical_pipeline', categorical_pipeline, categorical_columns)
  ])
  
  return processor

def instantiate_model(trial : Trial, numerical_columns : List[str], categorical_columns : List[str]) -> Pipeline:
  
  processor = instantiate_processor(
    trial, numerical_columns, categorical_columns
  )
  extra_trees = instantiate_extra_trees(trial)
  
  model = Pipeline([
    ('processor', processor),
    ('extra_trees', extra_trees)
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

study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=100)

study.best_params

best_trial = study.best_trial

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
print(score)

model.fit(X_full, y_full)

