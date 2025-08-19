from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import root_mean_squared_error, r2_score

from pyspark.sql import DataFrame
import mlflow
import pandas as pd 
from mlflow.models.signature import infer_signature
from src.utils import feature_label

# ======================
# Modeling Tune HyperParameters (sklearn)
# ======================

def apply_randomforest_with_RandomizedSearchCV(df_train_X: pd.DataFrame, train_y: pd.Series, seed=42):

    RF = RandomForestRegressor(random_state=seed)

    parameters = [{'n_estimators' : [20, 40, 60, 80],
                    'max_depth'    : [10, 20, 30, 40, 50],
                    'max_features' : [0.9, 0.7, 0.5],
                    'min_samples_split': [2, 4, 6],
                    }]

    grid_RF = RandomizedSearchCV(estimator=RF, param_distributions= parameters, cv = 5, n_jobs=-1)
    grid_RF.fit(df_train_X, train_y)

    print(" Results from Randomized Search " )
    print("\n The best estimator across ALL searched params:\n",grid_RF.best_estimator_)
    print("\n The best score across ALL searched params:\n",grid_RF.best_score_)
    print("\n The best parameters across ALL searched params:\n",grid_RF.best_params_)

    best_estimator = grid_RF.best_estimator_
    best_estimator.fit(df_train_X, train_y)
    train_predictions = best_estimator.predict(df_train_X)

    rmse_train = root_mean_squared_error(train_y, train_predictions)
    print("training rmse:", rmse_train)
    train_r2 = r2_score(train_y, train_predictions)
    print(train_r2)

    return grid_RF.best_estimator_, grid_RF.best_params_, grid_RF.best_score_


def apply_XGBBoosting_with_RandomizedSearchCV(df_train_X: pd.DataFrame, train_y: pd.Series, seed=42):
    param_grid = {
        'max_depth': [3, 5, 7, 10, 15],
        'learning_rate': [0.01, 0.03, 0.06, 0.1],
        'n_estimators': [200, 300, 500],
        'gamma': [0, 1, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0, 1, 10],
        'reg_lambda': [0, 1, 10]
    }

    search = RandomizedSearchCV(
        XGBRegressor(objective='reg:squarederror', random_state=seed),
        param_distributions=param_grid,
        n_iter=30,
        scoring='neg_root_mean_squared_error',
        cv=5,
        verbose=1,
        n_jobs=-1
    )
    
    search.fit(df_train_X, train_y)

    best_estimator = search.best_estimator_
    best_estimator.fit(df_train_X, train_y)
    train_predictions = best_estimator.predict(df_train_X)

    rmse_train = root_mean_squared_error(train_y, train_predictions)
    print("training rmse:", rmse_train)
    train_r2 = r2_score(train_y, train_predictions)
    print(train_r2)

    return search.best_estimator_, search.best_params_, search.best_score_


def split_choose_data(spark_df:DataFrame, seed=42):
    # 4. Preprocess first, then split
    # Here we split the original data into train/val/test AFTER preprocessing
    train_df_spark, val_df_spark, test_df_spark = spark_df.randomSplit([0.7, 0.15, 0.15], seed=seed)
    
    # convert it to Pandas dataframe for training and verification
    train_df = train_df_spark.select(feature_label.FEATURE_COLS + [feature_label.LABEL_COL]).toPandas()
    val_df = val_df_spark.select(feature_label.FEATURE_COLS + [feature_label.LABEL_COL]).toPandas()
    test_df = test_df_spark.select(feature_label.FEATURE_COLS + [feature_label.LABEL_COL]).toPandas()
    
    return train_df, val_df, test_df

# ======================
# Training entrypoints
# ======================

def train_with_model(spark_df: DataFrame, model_type ="xgb", seed=42):
    
    train_df, val_df, test_df = split_choose_data(spark_df)
    # Split vector column into separate features

    y_train = train_df[feature_label.LABEL_COL]
    X_train = train_df.drop(feature_label.LABEL_COL, axis = 1)
    
    y_val = val_df[feature_label.LABEL_COL]
    X_val = val_df.drop(feature_label.LABEL_COL, axis=1)
    
    y_test = test_df[feature_label.LABEL_COL]
    X_test = test_df.drop(feature_label.LABEL_COL, axis=1)

    
    # train the model
    print("train the model and choose the best parameters ...")
    if model_type.lower() == "rf":
        best_model, best_params, cv_rmse = apply_randomforest_with_RandomizedSearchCV(X_train, y_train, seed=seed)
        mlflow_model_logger = mlflow.sklearn.log_model
        model_flavor = "random_forest"
    else:
        best_model, best_params, cv_rmse = apply_XGBBoosting_with_RandomizedSearchCV(X_train, y_train, seed=seed)
        # xgboost flavor keeps booster + conda env; works well on Databricks
        mlflow_model_logger = mlflow.xgboost.log_model
        model_flavor = "xgboost"

    train_predictions = best_model.predict(X_train)
    val_predictions = best_model.predict(X_val)
    test_predictions = best_model.predict(X_test)
        
    # Metrics
    rmse_train = root_mean_squared_error(y_train, train_predictions)
    rmse_val = root_mean_squared_error(y_val, val_predictions)
    rmse_test = root_mean_squared_error(y_test, test_predictions)

    r2_train = r2_score(y_train, train_predictions)
    r2_val = r2_score(y_val, val_predictions)
    r2_test = r2_score(y_test, test_predictions)

    # metrics to be saved
    metrics = {
        "rmse_train": rmse_train,
        "rmse_val":   rmse_val,
        "rmse_test":  rmse_test,
        "r2_train":   r2_train,
        "r2_val":     r2_val,
        "r2_test":    r2_test,
    }
    
    parameters = {}
    for k, v in best_params.items():
        parameters[k] = v
        
    # parameters to be saved
    parameters["n_train"] = len(X_train)
    parameters["n_val"] = len(X_val)
    parameters["n_test"] = len(X_test)
    parameters["feature_count"] = len(feature_label.FEATURE_COLS)
    parameters["seed"] = seed
    parameters['model_type'] = model_flavor
    
    signature = infer_signature(X_train, y_train)

    # 3. Create input example
    input_example = X_train.head(10)
    input_example = mlflow.data.from_pandas(input_example)

    return best_model, parameters, metrics, signature, input_example
        