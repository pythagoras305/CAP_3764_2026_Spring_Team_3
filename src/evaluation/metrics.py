import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score
)


def evaluate_classifier(name: str, model, X_train, y_train, X_test, y_test,
                         threshold: float = 0.5):
    #Full evaluation of accuracy, confusion matrix, reports, and ROC-AUC

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")

    y_pred  = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    ap  = average_precision_score(y_test, y_proba)

    # Checking for Overfitting
    train_acc = accuracy_score(y_train, model.predict(X_train))
    gap = train_acc - acc
    print(f"Train Accuracy : {train_acc:.4f}")
    print(f"Test  Accuracy : {acc:.4f}")
    print(f"Gap (overfit)  : {gap:.4f}  {'⚠ possible overfit' if gap > 0.10 else '✓ ok'}")
    print(f"ROC-AUC        : {auc:.4f}")
    print(f"Avg Precision  : {ap:.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, digits=3)}")

    return {'model': name, 'train_acc': train_acc, 'test_acc': acc,
            'gap': gap, 'roc_auc': auc, 'avg_precision': ap}


def plot_roc_curve(models_dict: dict, X_test, y_test, title: str = "ROC Curves"):
    
    #Roc Curves plotted for multiple models
    
    plt.figure(figsize=(7, 5))
    for name, model in models_dict.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    plt.plot([0,1],[0,1], 'k--', label='Random baseline')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig('notebooks/figures/roc_curves.png', dpi=150)
    plt.show()


def plot_confusion_matrix(model, X_test, y_test, model_name: str = "XGBoost"):
    
    # Confusion Matrix Heatmap
    
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['DOWN', 'UP'], yticklabels=['DOWN', 'UP'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.savefig('notebooks/figures/confusion_matrix.png', dpi=150)
    plt.show()


def plot_feature_importance(model, feature_names: list, top_n: int = 10):
    """
    Bar chart of XGBoost feature importances.
    model must be a fitted XGBClassifier (not a Pipeline).
    """
    importance = model.feature_importances_
    idx = np.argsort(importance)[::-1][:top_n]
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importance[idx], y=np.array(feature_names)[idx], palette='Blues_r')
    plt.xlabel("Importance Score")
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.savefig('notebooks/figures/feature_importance.png', dpi=150)
    plt.show()


def backtest_simulation(model, X_test, y_test, test_df: pd.DataFrame):
    """
    Simple long-only backtest:
    Buy AAPL on days the model predicts UP, hold for 1 day.
    Compare cumulative return vs buy-and-hold benchmark.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)

    results = test_df[['date', 'daily_return']].copy().reset_index(drop=True)
    results['signal']          = y_pred
    results['strategy_return'] = results['signal'] * results['daily_return']

    results['cumulative_strategy']  = (1 + results['strategy_return']).cumprod()
    results['cumulative_buyhold']   = (1 + results['daily_return']).cumprod()

    # Summary stats
    total_strategy = results['cumulative_strategy'].iloc[-1] - 1
    total_buyhold  = results['cumulative_buyhold'].iloc[-1]  - 1
    signal_days    = results['signal'].sum()

    print(f"\n{'='*40}")
    print(f"  Backtest Results (Test Period)")
    print(f"{'='*40}")
    print(f"Strategy return  : {total_strategy*100:.1f}%")
    print(f"Buy-hold return  : {total_buyhold*100:.1f}%")
    print(f"Days traded      : {signal_days} / {len(results)}")

    plt.figure(figsize=(10, 5))
    plt.plot(results['date'], results['cumulative_strategy'], label='Sentiment Signal Strategy')
    plt.plot(results['date'], results['cumulative_buyhold'],  label='Buy & Hold', linestyle='--')
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.title("Strategy vs Buy-and-Hold (Test Period)")
    plt.legend()
    plt.tight_layout()
    plt.savefig('notebooks/figures/backtest.png', dpi=150)
    plt.show()

    return results
