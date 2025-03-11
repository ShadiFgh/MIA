import pandas as pd
from datasets import load_dataset
import printtextme
from printtextme import printTextme

def split_into_batches(df, batch_size):
    return [df[i:i + batch_size] for i in range(0, df.shape[0], batch_size)]

def get_dataset(num_lines=None, dataset_name=None, selected_batch=None):
    if not dataset_name:
        dataset_name = "ag_news"
    
    if not selected_batch:
        selected_batch = 0

    dataset = load_dataset(dataset_name)
    df_test = pd.DataFrame( dataset['test'] )
    df_test = df_test.dropna()
    df_train = pd.DataFrame( dataset['train'] )
    df_train = df_train.dropna()

    # Get the number of rows in each DataFrame
    num_rows_test = df_test.shape[0]
    num_rows_train = df_train.shape[0]

    # Find the smaller size
    min_rows = min(num_rows_test, num_rows_train)

    # Trim the larger DataFrame to match the size of the smaller one
    df_test = df_test.iloc[:min_rows, :]
    df_train = df_train.iloc[:min_rows, :]

    if num_lines:
        test_dataset_batches = split_into_batches(df_test, num_lines)
        train_dataset_batches = split_into_batches(df_train, num_lines)
        if selected_batch >= len(train_dataset_batches):
            selected_batch = len(train_dataset_batches) - 1
        selected_df_test = test_dataset_batches[selected_batch]
        selected_df_train = train_dataset_batches[selected_batch]
        return selected_df_train, selected_df_test

    return df_train, df_test
