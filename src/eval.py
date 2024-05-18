from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, f1_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

def evaluation_metrics(matrix_confusion):
  tn, fp, fn, tp = matrix_confusion.ravel()

  PPV = tp/(tp+fp)
  NPV = tn/(fn+tn)
  sensitivity = tp/(tp+fn)
  specificity = tn/(tn+fp)
  accuracy = (tp+tn)/(tp+fp+fn+tn)
  return {
      "PPV": PPV,
      "NPV": NPV,
      "sensitivity": sensitivity,
      "specificity": specificity,
      "accuracy": accuracy
  }

def metrics(y_true, y_pred):
  f1 = f1_score(y_true, y_pred)


def roc_curve(y_true, y_pred):
  
  fpr, tpr, thresholds = roc_curve(y_true, y_pred)
  auc = roc_auc_score(y_true, y_pred)
  plt.figure(figsize=(6, 6))
  plt.plot(fpr,tpr,label= f"AUC={auc}")
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('ROC Curve')
  plt.legend()
  plt.show()
  print()
  print('FPR = {}, TPR = {}'.format(fpr, tpr))