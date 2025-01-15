import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_correlation_heatmap(df: pd.DataFrame, figsize=(10, 8)):
    plt.figure(figsize=figsize)
    corr = df.corr()
    sns.heatmap(corr, cmap='coolwarm', annot=False)
    plt.title("Correlation Heatmap")
    plt.show()

def plot_predictions(y_true, y_pred, title="Predictions vs Actual",
                     xlabel="Time", ylabel="Price"):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true.index, y_true.values, label='Actual')
    plt.plot(y_true.index, y_pred, label='Predicted', alpha=0.7)
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_feature_importances(feature_importances, top_n=10):
    """
    Takes a pd.Series of feature importances, plot top n.
    """
    top_features = feature_importances.head(top_n)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=top_features, y=top_features.index)
    plt.title("Top Feature Importances")
    plt.show()
