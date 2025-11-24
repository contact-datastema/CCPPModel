import os
import sys

# Add the current directory to sys.path to allow imports if running directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from data_loader import load_data, split_data
    from model_trainer import train_and_evaluate
    from visualizer import plot_feature_vs_target, plot_correlation_heatmap, plot_residuals
except ImportError:
    # Fallback for when running as a module (python -m src.main)
    from src.data_loader import load_data, split_data
    from src.model_trainer import train_and_evaluate
    from src.visualizer import plot_feature_vs_target, plot_correlation_heatmap, plot_residuals

def main():
    # Define path to data relative to this script
    # Script is in src/, data is in data/ (sibling to src)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, 'data', 'CCPP_data.csv')
    
    # 1. Load Data
    print("Step 1: Loading Data...")
    df = load_data(data_path)
    if df is None:
        sys.exit(1)
        
    # 1b. Visualize Data
    print("\nStep 1b: Generating Data Visualizations...")
    plot_feature_vs_target(df)
    plot_correlation_heatmap(df)
        
    # 2. Split Data
    print("\nStep 2: Splitting Data...")
    X_train, X_test, y_train, y_test = split_data(df)
    
    # 3. Train, Compare and Evaluate
    print("\nStep 3: Training and Evaluating Models...")
    final_model, final_rmse, final_r2 = train_and_evaluate(X_train, y_train, X_test, y_test)
    
    # 4. Visualize Predictions
    print("\nStep 4: Visualizing Model Performance...")
    y_pred = final_model.predict(X_test)
    plot_residuals(y_test, y_pred)
    
    print("\nPipeline completed successfully.")

if __name__ == "__main__":
    main()
