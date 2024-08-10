import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_confusion_matrix(conf_matrix, classes, title='Confusion Matrix'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(title)
    plt.show()

def plot_classification_report(report_df, title='Classification Report'):
    plt.figure(figsize=(10, 8))
    sns.barplot(x='f1-score', y='class', data=report_df)
    plt.xlabel('F1-score')
    plt.ylabel('Class')
    plt.title(title)
    plt.show()

def plot_loss_curve(losses, title='Loss Curve'):
    plt.figure(figsize=(10, 8))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.show()

def plot_accuracy_curve(accuracies, title='Accuracy Curve'):
    plt.figure(figsize=(10, 8))
    plt.plot(accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.show()

def plot_roc_curve(y_true, y_pred, title='ROC Curve'):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.show()

def plot_feature_importances(importances, feature_names, title='Feature Importances'):
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(importances)), importances)
    plt.yticks(range(len(importances)), feature_names)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(title)
    plt.show()
