import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_feature_vs_target(df, target_column='PE', output_dir='plots'):
    """
    Generates scatter plots for each feature vs the target variable.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    features = [col for col in df.columns if col != target_column]
    
    for feature in features:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x=feature, y=target_column, alpha=0.5)
        plt.title(f'{feature} vs {target_column}')
        plt.xlabel(feature)
        plt.ylabel(target_column)
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{feature}_vs_{target_column}.png'))
        plt.close()
    
    print(f"Feature scatter plots saved to {output_dir}/")

def plot_correlation_heatmap(df, output_dir='plots'):
    """
    Generates a correlation heatmap for the dataframe.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()
    
    print(f"Correlation heatmap saved to {output_dir}/")

def plot_residuals(y_test, y_pred, output_dir='plots'):
    """
    Generates a residual plot (Actual vs Predicted).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    plt.title('Actual vs Predicted Energy Output')
    plt.xlabel('Actual PE (MW)')
    plt.ylabel('Predicted PE (MW)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'))
    plt.close()
    
    print(f"Residual plot saved to {output_dir}/")
