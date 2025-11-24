import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_and_evaluate(X_train, y_train, X_test, y_test):
    """
    Trains Linear Regression and Random Forest models.
    Performs Hyperparameter Tuning for Random Forest.
    Evaluates the best model on the test set.
    """
    
    # 1. Linear Regression (Baseline)
    print("\n--- Model 1: Linear Regression (Baseline) ---")
    lr_model = LinearRegression()
    cv_scores_lr = cross_val_score(lr_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    rmse_lr = np.sqrt(-cv_scores_lr).mean()
    print(f"Linear Regression CV RMSE: {rmse_lr:.4f}")

    # 2. Random Forest (Default)
    print("\n--- Model 2: Random Forest (Default) ---")
    rf_default = RandomForestRegressor(n_estimators=100, random_state=42)
    cv_scores_rf_default = cross_val_score(rf_default, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    rmse_rf_default = np.sqrt(-cv_scores_rf_default).mean()
    print(f"Random Forest (Default) CV RMSE: {rmse_rf_default:.4f}")

    # 3. Random Forest with Hyperparameter Tuning
    print("\n--- Model 3: Random Forest (Hyperparameter Tuning) ---")
    rf = RandomForestRegressor(random_state=42)
    
    # Define at least 3 scenarios (combinations of hyperparameters)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    print(f"Tuning Random Forest with grid: {param_grid}")
    
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_rf_model = grid_search.best_estimator_
    best_rf_rmse = np.sqrt(-grid_search.best_score_)
    
    print(f"\nBest Random Forest Parameters: {grid_search.best_params_}")
    print(f"Best Random Forest (Tuned) CV RMSE: {best_rf_rmse:.4f}")
    
    # Compare Default vs Tuned
    improvement = rmse_rf_default - best_rf_rmse
    print(f"\nImprovement over Default RF: {improvement:.4f} MW ({(improvement/rmse_rf_default)*100:.2f}%)")
    
    # Compare and Select Best Model
    # We compare LR, Default RF, and Tuned RF
    models_perf = {
        "Linear Regression": rmse_lr,
        "Random Forest (Default)": rmse_rf_default,
        "Random Forest (Tuned)": best_rf_rmse
    }
    
    best_model_name = min(models_perf, key=models_perf.get)
    print(f"\nOverall Best Model selected: {best_model_name}")
    
    if best_model_name == "Linear Regression":
        final_model = lr_model
        final_model.fit(X_train, y_train)
    elif best_model_name == "Random Forest (Default)":
        final_model = rf_default
        final_model.fit(X_train, y_train)
    else:
        final_model = best_rf_model # Already fitted by GridSearchCV

    # Evaluate on Test set
    y_pred = final_model.predict(X_test)
    final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    final_r2 = r2_score(y_test, y_pred)
    
    print(f"\n--- Final Evaluation on Test Set ({best_model_name}) ---")
    print(f"RMSE: {final_rmse:.4f} MW")
    print(f"R2 Score: {final_r2:.4f}")
    
    return final_model, final_rmse, final_r2
