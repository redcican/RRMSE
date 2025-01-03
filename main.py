import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import time
import psutil
import sys
from memory_profiler import profile
from functools import wraps

def preprocess_data(data):
    X = data.drop(data.columns[-1], axis=1)
    y = data[data.columns[-1]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred)
    }
    
def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        metrics = {
            'execution_time': end_time - start_time,
            'memory_usage': end_memory - start_memory,
        }
        return result, metrics
    return wrapper

@measure_time
def train_and_evaluate_model_overhead(model, X_train, y_train, X_test, y_test):
    # Training time
    train_start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - train_start
    
    # Prediction time
    predict_start = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - predict_start
    
    metrics = evaluate_model(model, X_test, y_test)
    metrics.update({
        'train_time': train_time,
        'predict_time': predict_time
    })
    
    return metrics

# Base learners search spaces
base_search_spaces = {
    'linear': {
        'fit_intercept': Categorical([True, False]),
        'normalize': Categorical([True, False])
    },
    'knn': {
        'n_neighbors': Integer(1, 50),
        'weights': Categorical(['uniform', 'distance']),
        'leaf_size': Integer(10, 50),
        'p': Integer(1, 3)
    },
    'sgd': {
        'alpha': Real(1e-6, 1e-1, prior='log-uniform'),
        'l1_ratio': Real(0, 1),
        'loss': Categorical(['squared_error', 'huber']),
        'penalty': Categorical(['l2', 'l1', 'elasticnet']),
        'learning_rate': Categorical(['constant', 'optimal', 'adaptive'])
    },
    'rf': {
        'n_estimators': Integer(50, 300),
        'max_depth': Integer(3, 15),
        'min_samples_split': Integer(2, 20),
        'min_samples_leaf': Integer(1, 10),
        'max_features': Categorical(['auto', 'sqrt', 'log2'])
    }
}

# Extended search spaces for advanced boosting methods
advanced_search_spaces = {
    'xgb': {
        'learning_rate': Real(0.01, 0.3),
        'max_depth': Integer(3, 12),
        'n_estimators': Integer(50, 400),
        'min_child_weight': Integer(1, 7),
        'subsample': Real(0.6, 1.0),
        'colsample_bytree': Real(0.6, 1.0),
        'gamma': Real(0, 5),
        'reg_alpha': Real(0, 5),
        'reg_lambda': Real(1, 5)
    },
    'lgb': {
        'learning_rate': Real(0.01, 0.3),
        'num_leaves': Integer(20, 150),
        'n_estimators': Integer(50, 400),
        'min_child_samples': Integer(10, 100),
        'subsample': Real(0.6, 1.0),
        'colsample_bytree': Real(0.6, 1.0),
        'reg_alpha': Real(0, 5),
        'reg_lambda': Real(0, 5),
        'min_split_gain': Real(0, 1)
    },
    'cat': {
        'learning_rate': Real(0.01, 0.3),
        'depth': Integer(3, 12),
        'iterations': Integer(50, 400),
        'l2_leaf_reg': Real(1, 10),
        'border_count': Integer(32, 255),
        'bagging_temperature': Real(0, 1),
        'random_strength': Real(1, 20),
        'min_data_in_leaf': Integer(1, 50)
    }
}

def optimize_base_learners(X_train, y_train):
    optimized_models = {}
    
    lr = BayesSearchCV(
        LinearRegression(),
        base_search_spaces['linear'],
        n_iter=10,
        cv=5,
        n_jobs=-1
    )
    
    # KNN optimization
    knn = BayesSearchCV(
        KNeighborsRegressor(),
        base_search_spaces['knn'],
        n_iter=20,
        cv=5,
        n_jobs=-1
    )
    
    # SGD optimization
    sgd = BayesSearchCV(
        SGDRegressor(max_iter=1000),
        base_search_spaces['sgd'],
        n_iter=20,
        cv=5,
        n_jobs=-1
    )
    
    # Random Forest optimization
    rf = BayesSearchCV(
        RandomForestRegressor(),
        base_search_spaces['rf'],
        n_iter=20,
        cv=5,
        n_jobs=-1
    )
    
    # Fit all base models
    for name, model in zip(['linear','knn', 'sgd', 'rf'], [lr, knn, sgd, rf]):
        model.fit(X_train, y_train)
        optimized_models[name] = model.best_estimator_
    
    return optimized_models

def optimize_advanced_learners(X_train, y_train):
    optimized_models = {}
    
    # XGBoost optimization
    xgb = BayesSearchCV(
        XGBRegressor(),
        advanced_search_spaces['xgb'],
        n_iter=30,
        cv=5,
        n_jobs=-1
    )
    
    # LightGBM optimization
    lgb = BayesSearchCV(
        LGBMRegressor(),
        advanced_search_spaces['lgb'],
        n_iter=30,
        cv=5,
        n_jobs=-1
    )
    
    # CatBoost optimization
    cat = BayesSearchCV(
        CatBoostRegressor(verbose=False),
        advanced_search_spaces['cat'],
        n_iter=30,
        cv=5,
        n_jobs=-1
    )
    
    # Fit all advanced models
    for name, model in zip(['xgb', 'lgb', 'cat'], [xgb, lgb, cat]):
        model.fit(X_train, y_train)
        optimized_models[name] = model.best_estimator_
    
    return optimized_models


def main():
    # Load datasets
    datasets = {
        'abalone': pd.read_csv('data/abalone.csv'),
        'Airfoil': pd.read_csv('data/Airfoil.csv'),
        'car': pd.read_csv('data/car.csv'),
        'diamonds': pd.read_csv('data/diamonds.csv'),
        'Elongation': pd.read_csv('data/Elongation.csv'),
        'smart_grid__stability': pd.read_csv('data/smart_grid__stability.csv')
    }
    
    results = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    overhead_metrics = {
        dataset_name: {
            model_name: {
                'avg_train_time': [],
                'avg_predict_time': [],
                'avg_memory_usage': [],
            } for model_name in optimized_models.keys()
        } for dataset_name in datasets.keys()
    }
    
    for dataset_name, data in datasets.items():
        X_scaled, y = preprocess_data(data)
        dataset_results = []
        
        for train_idx, test_idx in kf.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            optimized_models = optimize_base_learners(X_train, y_train)
            fold_results = {}
            
            for model_name, model in optimized_models.items():
                metrics = evaluate_model(model, X_test, y_test)
                fold_results[model_name] = metrics
                
                _, overhead = train_and_evaluate_model_overhead(
                    model, X_train, y_train, X_test, y_test
                )
                
                overhead_metrics[dataset_name][model_name]['avg_train_time'].append(overhead['train_time'])
                overhead_metrics[dataset_name][model_name]['avg_predict_time'].append(overhead['predict_time'])
                overhead_metrics[dataset_name][model_name]['avg_memory_usage'].append(overhead['memory_usage'])
            
            dataset_results.append(fold_results)
            dataset_results.append(overhead_metrics)
            
        results[dataset_name] = dataset_results

if __name__ == "__main__":
    main()
