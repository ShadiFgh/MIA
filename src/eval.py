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
  print()
  print('FPR = {}, TPR = {}'.format(fpr, tpr))
  print()
  print('AUC = {}'.format(auc))


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