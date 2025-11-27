import numpy as np

def get_confusion_matrix(y_hat, y_true, num_classes):
    """
    Computes the confusion matrix for multiple classes.
    Rows: Actual, Columns: Predicted
    """
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(len(y_hat)):
        confusion_matrix[int(y_true[i])][int(y_hat[i])] += 1
    return confusion_matrix

def precision(confusion_matrix, idx):
    """
    Calculates Precision for a specific class index.
    Formula: TP / (TP + FP)
    """
    col_sum = confusion_matrix[:, idx].sum()
    if col_sum == 0:
        return 0.0
    return confusion_matrix[idx][idx] / col_sum

def recall(confusion_matrix, idx):
    """
    Calculates Recall for a specific class index.
    Formula: TP / (TP + FN)
    """
    row_sum = confusion_matrix[idx, :].sum()
    if row_sum == 0:
        return 0.0
    return confusion_matrix[idx][idx] / row_sum

def f1_score(p, r):
    """
    Calculates F1 Score (Harmonic mean of Precision and Recall).
    """
    if (p + r) == 0:
        return 0.0
    return 2 * ((p * r) / (p + r))

def accuracy(confusion_matrix):
    """
    Calculates global accuracy.
    Formula: Total Correct / Total Samples
    """
    total = confusion_matrix.sum()
    if total == 0: return 0.0
    return confusion_matrix.diagonal().sum() / total

def get_classification_report(y_hat, y_true, num_classes):
    """
    Generates a summary of classification metrics.
    Returns: accuracy, recall, precision, f1_score
    Note: Returns Macro Average for multiclass > 2.
    """
    confusion_matrix = get_confusion_matrix(y_hat, y_true, num_classes)
    acc = accuracy(confusion_matrix)

    if num_classes > 2:
        r_list = np.zeros((num_classes))
        p_list = np.zeros((num_classes))
        f1_list = np.zeros((num_classes))

        for i in range(num_classes):
            r_list[i] = recall(confusion_matrix, i)
            p_list[i] = precision(confusion_matrix, i)
            f1_list[i] = f1_score(p_list[i], r_list[i])
        
        r = r_list.mean()
        p = p_list.mean()
        f1 = f1_list.mean()
    else:
        r = recall(confusion_matrix, 1)
        p = precision(confusion_matrix, 1)
        f1 = f1_score(p, r)

    return acc, r, p, f1


def mae(y_hat, y_true):
    """
    Mean Absolute Error: Average absolute difference.
    Robust to outliers.
    """
    return np.mean(np.abs(y_true - y_hat))

def mse(y_hat, y_true):
    """
    Mean Squared Error: Average squared difference.
    Punishes larger errors heavily.
    """
    return np.mean(np.power(y_true - y_hat, 2))

def rmse(y_hat, y_true):
    """
    Root Mean Squared Error: Square root of MSE.
    """
    return np.sqrt(mse(y_hat, y_true))

def r2(y_hat, y_true):
    """
    R-Squared (Coefficient of Determination).
    1.0 is perfect, 0.0 is baseline (mean guessing).
    """
    ss_res = np.sum(np.power(y_true - y_hat, 2))
    ss_tot = np.sum(np.power(y_true - y_true.mean(), 2))
    
    if ss_tot == 0: return 0.0
    return 1 - (ss_res / ss_tot)

def get_regression_report(y_hat, y_true):
    """
    Generates a summary of regression metrics.
    Returns: MAE, MSE, RMSE, R2
    """
    mae_val = mae(y_hat, y_true)
    mse_val = mse(y_hat, y_true)
    rmse_val = rmse(y_hat, y_true)
    r2_val = r2(y_hat, y_true)
    
    return mae_val, mse_val, rmse_val, r2_val

def print_classification_report(acc, r, p, f1, num_classes):
    """
    Prints a formatted table for classification metrics.
    """
    mode = "Binary (Class 1)" if num_classes == 2 else "Multiclass (Macro Avg)"
    
    print("\n" + "="*40)
    print(f"  CLASSIFICATION REPORT  ")
    print("="*40)
    print(f"Mode      : {mode}")
    print("-" * 40)
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {p:.4f}")
    print(f"Recall    : {r:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print("="*40 + "\n")

def print_regression_report(mae, mse, rmse, r2):
    """
    Prints a formatted table for regression metrics.
    """
    print("\n" + "="*40)
    print(f"  REGRESSION REPORT  ")
    print("="*40)
    print(f"MAE       : {mae:.4f}")
    print(f"MSE       : {mse:.4f}")
    print(f"RMSE      : {rmse:.4f}")
    print(f"R-Squared : {r2:.4f}")
    print("="*40 + "\n")