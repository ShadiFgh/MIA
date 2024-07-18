import pandas as pd
from datasets import load_dataset

RESULT_SAVE_PATH = "Result"

def printTextShadi(*args, **kwargs):
    with open(f'{RESULT_SAVE_PATH}/output.txt', 'a') as f:
        for arg in args:
            print(arg)
            f.write(f"{arg}\n")
        for key, value in kwargs.items():
            print(f"{key}: {value}")
            f.write(f"{key}: {value}\n")

def get_dataset(num_lines=None):
  dataset = load_dataset('ag_news')
  df_test = pd.DataFrame( dataset['test'] )
  df_test = df_test.dropna()
  df_train = pd.DataFrame( dataset['train'] )
  df_train = df_train.dropna()

  if num_lines:
    return df_train[0:num_lines], df_test[0:num_lines]
  return df_train, df_test