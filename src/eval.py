from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, f1_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

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