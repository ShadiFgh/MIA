# from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, f1_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

def printTextShadi(*args, **kwargs):
    with open('output.txt', 'a') as f:
        for arg in args:
            print(arg)
            f.write(f"{arg}\n")
        for key, value in kwargs.items():
            print(f"{key}: {value}")
            f.write(f"{key}: {value}\n")

def confusion_matrix(y_true, y_pred):
    """
    Calculate the elements of the confusion matrix: True Positives (TP), 
    False Positives (FP), False Negatives (FN), and True Negatives (TN).

    Parameters:
    y_true (list or array): List or array of true binary labels (0 or 1).
    y_pred (list or array): List or array of predicted binary labels (0 or 1).

    Returns:
    tp (int): True Positives
    fp (int): False Positives
    fn (int): False Negatives
    tn (int): True Negatives
    """
    tp = sum((y_true[i] == 1) and (y_pred[i] == 1) for i in range(len(y_true)))
    fp = sum((y_true[i] == 0) and (y_pred[i] == 1) for i in range(len(y_true)))
    fn = sum((y_true[i] == 1) and (y_pred[i] == 0) for i in range(len(y_true)))
    tn = sum((y_true[i] == 0) and (y_pred[i] == 0) for i in range(len(y_true)))
    
    return tp, fp, fn, tn

def evaluation_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred)
    printTextShadi(f"TP: {tp}\nFP: {fp}\nFN: {fn}\nTN: {tp}")
    PPV, NPV, sensitivity, specificity, accuracy = None, None, None, None, None
    if tp+fp > 0:
        PPV = tp/(tp+fp)
    else:
        printTextShadi("PPV is undefined, since TP+FP=0")
    if tn+fn > 0:
        NPV = tn/(fn+tn)
    else:
        printTextShadi("NPV is undefined, since TN+FN=0")
    if tp+fn > 0:
        sensitivity = tp/(tp+fn)
    else:
        printTextShadi("Sensitivity is undefined, since TP+FN=0")
    if tn+fp > 0:
        specificity = tn/(tn+fp)
    else:
        printTextShadi("Specificity is undefined, since TN+FP=0")
    if tp+fp+fn+tn > 0:
        accuracy = (tp+tn)/(tp+fp+fn+tn)
    else:
        printTextShadi("Accuracy is undefined, since TP+FP+FN+TN=0") 
    f1 = f1_score(y_true, y_pred, pos_label=1)
    return {
        "Precision (Positive Predictive Value, PPV)": PPV,
        "NPV": NPV,
        "Recall or Sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": accuracy,
        "f1 score": f1
    }

def eval_roc_curve(y_true, y_pred):

    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    auc = roc_auc_score(y_true, y_pred)
    plt.figure()
    plt.plot(fpr,tpr,label= f"AUC={auc}")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
    printTextShadi()
    printTextShadi('FPR = {}, TPR = {}'.format(fpr, tpr))
    printTextShadi()
    printTextShadi('AUC = {}'.format(auc))


# Function to interpolate TPR at specific FPR
def interpolate_tpr(fpr, tpr, target_fpr):
    return np.interp(target_fpr, fpr, tpr)


def calculate_tpr_at_fpr(y_true, y_pred):
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    # TPR at 2%, 5%, and 10% FPR
    tpr_at_2_fpr = interpolate_tpr(fpr, tpr, 0.02)
    tpr_at_5_fpr = interpolate_tpr(fpr, tpr, 0.05)
    tpr_at_10_fpr = interpolate_tpr(fpr, tpr, 0.10)

    return tpr_at_2_fpr, tpr_at_5_fpr, tpr_at_10_fpr