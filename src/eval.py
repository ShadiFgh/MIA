from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, f1_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

def evaluation_metrics(y_true, y_pred):
  
  tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[1, 0]).ravel()

  PPV = tp/(tp+fp)
  NPV = tn/(fn+tn)
  sensitivity = tp/(tp+fn)
  specificity = tn/(tn+fp)
  accuracy = (tp+tn)/(tp+fp+fn+tn)
  f1 = f1_score(y_true, y_pred, pos_label=1)
  return {
      "PPV": PPV,
      "NPV": NPV,
      "sensitivity": sensitivity,
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
  print()
  print('FPR = {}, TPR = {}'.format(fpr, tpr))
  print()
  print('AUC = {}'.format(auc))

  def evaluate_tpr_fpr_precision_recall_at_fixed_fpr(y_true, y_pred, fixed_fpr=0.01):
    """
    Evaluate TPR and Precision and Recall at a fixed FPR.

    Parameters:
    y_true (array-like): True binary labels (0 or 1).
    y_pred (array-like): Scores or probabilities for the positive class.
    fixed_fpr (float): The fixed false positive rate threshold.

    Returns:
    dict: Dictionary containing TPR, Precision, and Recall at the fixed FPR.
    """
    # Calculate ROC curve
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_pred)

    # Find the threshold for the fixed FPR
    threshold_index = np.where(fpr >= fixed_fpr)[0][0]
    threshold = thresholds_roc[threshold_index]

    # Calculate Precision-Recall curve
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_pred)

    # Find the closest threshold in the PR curve to the ROC threshold
    pr_threshold_index = np.argmin(np.abs(thresholds_pr - threshold))

    # Extract the TPR, Precision, and Recall at the fixed FPR threshold
    tpr_at_fixed_fpr = tpr[threshold_index]
    precision_at_fixed_fpr = precision[pr_threshold_index]
    recall_at_fixed_fpr = recall[pr_threshold_index]

    return {
        'TPR_at_fixed_FPR': tpr_at_fixed_fpr,
        'Precision_at_fixed_FPR': precision_at_fixed_fpr,
        'Recall_at_fixed_FPR': recall_at_fixed_fpr
    }