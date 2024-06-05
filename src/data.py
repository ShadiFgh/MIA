import pandas as pd
from datasets import load_dataset



def get_dataset():

  dataset = load_dataset('ag_news')
  df_test = pd.DataFrame( dataset['test'] )
  df_test = df_test.dropna()
  df_train = pd.DataFrame( dataset['train'] )
  df_train = df_train.dropna()

  return df_train[0:50], df_test[0:50]