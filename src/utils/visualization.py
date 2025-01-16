import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

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

def plot_time_series(y_true, y_pred=None, title="Time Series Plot",
                     time=None, xlabel="Time Steps", ylabel="Value",
                     figsize=(10, 5), show=True):
    """
    Plots a (time) series, optionally with predicted values.

    Parameters
    ----------
    y_true : array-like
        Ground truth or original series.
    y_pred : array-like, optional
        Predicted or second series (if you want to show it).
    time : array-like, optional
        The x-axis time or index. If None, range(len(y_true)) is used.
    title : str, optional
        Plot title.
    xlabel : str, optional
        Label for x-axis.
    ylabel : str, optional
        Label for y-axis.
    figsize : tuple, optional
        Figure size (width, height).
    show : bool, optional
        Whether to call plt.show() at the end.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    matplotlib.axes.Axes
        The axes object.

    """
    if time is None:
        time = range(len(y_true))

    y_true_np = np.array(y_true)
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(time, y_true_np, label="Actual", color='blue', linestyle='--')

    if y_pred is not None:
        y_pred_np = np.array(y_pred)
        ax.plot(time, y_pred_np, label="Predicted", color='red')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    if show:
        plt.show()

    return fig, ax

def plot_residuals(y_true, y_pred, title="Residuals Plot",
                   xlabel="Predicted", ylabel="Residuals",
                   figsize=(6, 5), show=True):
    """
    Plots residuals (y_true - y_pred) vs. predicted values.

    Parameters
    ----------
    y_true : array-like
        Ground truth.
    y_pred : array-like
        Predicted values from the model.
    title : str, optional
        Plot title.
    xlabel : str, optional
        Label for x-axis.
    ylabel : str, optional
        Label for y-axis.
    figsize : tuple, optional
        Figure size.
    show : bool, optional
        Whether to call plt.show() at the end.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    matplotlib.axes.Axes
        The axes object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    residuals = y_true_np - y_pred_np
    ax.scatter(y_pred_np, residuals, alpha=0.5, color='green')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if show:
        plt.show()

    return fig, ax

