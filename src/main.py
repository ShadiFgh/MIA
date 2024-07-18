import attack
import data
import eval
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import DataLoader
from target_model import GPT2Dataset
import target_model
import pandas as pd
import numpy as np
import sys
import torch
import time
import pickle
import os
from datetime import datetime

device_type = 'cpu' # cuda or cpu
device_id = 0

LOAD_DATA_FRAME = False
LOAD_MODEL = False

RESULT_SAVE_PATH = "Result"

# Get the current datetime
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

if not LOAD_DATA_FRAME and not LOAD_MODEL:
    # Create the new save path with the datetime appended
    RESULT_SAVE_PATH = f"{RESULT_SAVE_PATH}_{current_datetime}"

# Check if the path exists, and create it if it doesn't
if not os.path.exists(RESULT_SAVE_PATH):
    os.makedirs(RESULT_SAVE_PATH)

# Empty Output File
with open(f'{RESULT_SAVE_PATH}/output.txt', 'w') as f:
    f.write('')

def printTextShadi(*args, **kwargs):
    with open(f'{RESULT_SAVE_PATH}/output.txt', 'a') as f:
        for arg in args:
            print(arg)
            f.write(f"{arg}\n")
        for key, value in kwargs.items():
            print(f"{key}: {value}")
            f.write(f"{key}: {value}\n")

start_time = time.time()
printTextShadi(f'Started Program at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')

if len(sys.argv) > 1:
    if len(sys.argv) > 2:
        device_type = sys.argv[2].lower().strip()
    if len(sys.argv) == 4:
        device_id = int(sys.argv[3].lower().strip())
    if str(sys.argv[1]).lower().strip() == "testing":
        dataset_size = 3
    else:
        try:
            dataset_size = int(sys.argv[1])
        except:
            printTextShadi("Failed to parse number of lines of dataset you want to use!")
            printTextShadi("Example for testing that is 3 lines for default, Use like: python main.py testing")
            printTextShadi("Example for using 981 lines, Use like: python main.py 981")
            sys.exit()
    dataframe_train, dataframe_test = data.get_dataset(num_lines = dataset_size)
else:
    dataframe_train, dataframe_test = data.get_dataset()

printTextShadi(f"Device Type: {device_type}\nDevice ID:{device_id}\nDataset Size: {dataset_size}\n=========================================================================")

# List of available devices in array
devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
device = None
if device_type == "cpu":
    device = torch.device("cpu")
else:
    if torch.cuda.is_available():
        if device_id >= len(devices):
            printTextShadi(f"CUDA available but the Device with ID {device_id} is not available. Using the last available device.")
            device_id = -1
        device = devices[device_id]
    else:
        printTextShadi("CUDA is not available. Using CPU instead.")
        device = torch.device("cpu")

dataframe_train['y_true'] = 1
dataframe_test['y_true'] = 0
dataframe = pd.concat([dataframe_train, dataframe_test], ignore_index=True)
# deep copy of original dataframe
original_df_copy = dataframe.copy()
log_lines = []
if LOAD_DATA_FRAME and os.path.exists(f'{RESULT_SAVE_PATH}/dataframe_backtr1.csv'):
    if not os.path.exists(f'{RESULT_SAVE_PATH}/dataframe_backtr2.csv') and not os.path.exists(f'{RESULT_SAVE_PATH}/dataframe_backtr3.csv'):
        dataframe = pd.read_csv(f'{RESULT_SAVE_PATH}/dataframe_backtr1.csv')
        log_lines.append("Loaded Backtranslation 1 from dataframe_backtr1.csv")
        printTextShadi("Loaded Backtranslation 1 from dataframe_backtr1.csv")
else:
    dataframe['back_tr_1'] = attack.get_back_translations(dataframe['text'], 'spa_Latn', device=device)
    log_lines.append("Completed Backtranslation 1")
    printTextShadi("Completed Backtranslation 1")
    # Save dataframe for backtr1
    dataframe.to_csv(f'{RESULT_SAVE_PATH}/dataframe_backtr1.csv', index=False, header=True)
    log_lines.append("Saved Backtranslation 1 to dataframe_backtr1.csv")
    printTextShadi("Saved Backtranslation 1 to dataframe_backtr1.csv")

if LOAD_DATA_FRAME and os.path.exists(f'{RESULT_SAVE_PATH}/dataframe_backtr2.csv'):
    if not os.path.exists(f'{RESULT_SAVE_PATH}/dataframe_backtr3.csv'):
        dataframe = pd.read_csv(f'{RESULT_SAVE_PATH}/dataframe_backtr2.csv')
        log_lines.append("Loaded Backtranslation 2 from dataframe_backtr2.csv")
        printTextShadi("Loaded Backtranslation 2 from dataframe_backtr2.csv")
else:
    dataframe['back_tr_2'] = attack.get_back_translations(dataframe['back_tr_1'], 'fra', device=device)
    log_lines.append("Completed Backtranslation 2")
    printTextShadi("Completed Backtranslation 2")
    # Save dataframe for backtr2
    dataframe.to_csv(f'{RESULT_SAVE_PATH}/dataframe_backtr2.csv', index=False, header=True)
    log_lines.append("Saved Backtranslation 2 to dataframe_backtr2.csv")
    printTextShadi("Saved Backtranslation 2 to dataframe_backtr2.csv")

if LOAD_DATA_FRAME and os.path.exists(f'{RESULT_SAVE_PATH}/dataframe_backtr3.csv'):
    dataframe = pd.read_csv(f'{RESULT_SAVE_PATH}/dataframe_backtr3.csv')
    log_lines.append("Loaded Backtranslation 3 from dataframe_backtr3.csv")
    printTextShadi("Loaded Backtranslation 3 from dataframe_backtr3.csv")
else:
    dataframe['back_tr_3'] = attack.get_back_translations(dataframe['back_tr_2'], 'spa', device=device)
    log_lines.append("Completed Backtranslation 3")
    printTextShadi("Completed Backtranslation 3")
    # Save dataframe for backtr3
    dataframe.to_csv(f'{RESULT_SAVE_PATH}/dataframe_backtr3.csv', index=False, header=True)
    log_lines.append("Saved Backtranslation 3 to dataframe_backtr3.csv")
    printTextShadi("Saved Backtranslation 3 to dataframe_backtr3.csv")

# if dataframe is not having any row then raise an error
if dataframe.shape[0] == 0:
    raise ValueError("Dataframe is empty")
# If all lines of dataframe are same as original dataframe then raise an error
if dataframe.equals(original_df_copy):# and 1==2:
    raise ValueError("Dataframe is same as original dataframe")


tokenizer = GPT2Tokenizer.from_pretrained("gpt2", torch_dtype=torch.float32)
tg_model = None
if LOAD_MODEL and os.path.exists(f'{RESULT_SAVE_PATH}/trained_model.pkl'):
    # Initialize the model
    tg_model = GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=torch.float32)
    # Load the state dictionary
    with open(f'{RESULT_SAVE_PATH}/trained_model.pkl', 'rb') as f:
        state_dict = pickle.load(f)
    # Load the state dictionary into the model
    tg_model.load_state_dict(state_dict)
    # If you want to use it on the same device
    tg_model.to(device)
    log_lines.append("Loaded the Model State from trained_model.pkl")
    printTextShadi("Loaded the Model State from trained_model.pkl")
else:
    # Initialize the model and tokenizer
    tg_model = GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=torch.float32)
    tg_model.to(device)
    # Instantiate dataset class `GPT2Dataset`
    dataset = GPT2Dataset(dataframe_train['text'])
    dataloader = DataLoader(dataset, shuffle=False, batch_size=1)
    # Train the model
    target_model.trg_mdl_train(target_model=tg_model, dataloader=dataloader, device=device)
    # Save the model using pickle
    with open(f'{RESULT_SAVE_PATH}/trained_model.pkl', 'wb') as f:
        pickle.dump(tg_model.state_dict(), f)
    log_lines.append("Saved the Model State to trained_model.pkl")
    printTextShadi("Saved the Model State to trained_model.pkl")

if tg_model == None:
    raise Exception("Model is not loaded or is also not trained")

printTextShadi(f'Finished Training at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')

# tg_model = GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=torch.float16)
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2", torch_dtype=torch.float16)

dataset = GPT2Dataset(dataframe['text'])
dataloader = DataLoader(dataset, shuffle=False)
generated_text_df = pd.DataFrame()
eval_loss_df = pd.DataFrame()
tokens_df = pd.DataFrame()
resp = target_model.generate_text(tg_model, dataloader, tokenizer, device=device)
generated_text_df['original'] = resp[0]
eval_loss_df['original'] = [ten.cpu().numpy() for ten in resp[1]]
tokens_df['original'] = [ten.cpu().numpy() for ten in resp[2]]

dataset_back_tr_1 = GPT2Dataset(dataframe['back_tr_1'])
dataloader = DataLoader(dataset_back_tr_1, shuffle=False, batch_size=1)
resp2= target_model.generate_text(tg_model, dataloader, tokenizer, device=device)
generated_text_df['tr_1'] = resp2[0]
eval_loss_df['tr_1'] = [ten.cpu().numpy() for ten in resp2[1]]
tokens_df['tr_1'] = [ten.cpu().numpy() for ten in resp2[2]]

dataset_back_tr_2 = GPT2Dataset(dataframe['back_tr_2'])
dataloader = DataLoader(dataset_back_tr_2, shuffle=False, batch_size=1)
resp3= target_model.generate_text(tg_model, dataloader, tokenizer, device=device)
generated_text_df['tr_2'] = resp3[0]
eval_loss_df['tr_2'] = [ten.cpu().numpy() for ten in resp3[1]]
tokens_df['tr_2'] = [ten.cpu().numpy() for ten in resp3[2]]

dataset_back_tr_3 = GPT2Dataset(dataframe['back_tr_3'])
dataloader = DataLoader(dataset_back_tr_3, shuffle=False, batch_size=1)
resp4= target_model.generate_text(tg_model, dataloader, tokenizer, device=device)
generated_text_df['tr_3'] = resp4[0]
eval_loss_df['tr_3'] = [ten.cpu().numpy() for ten in resp4[1]]
tokens_df['tr_3'] = [ten.cpu().numpy() for ten in resp4[2]]

dataframe.to_csv(f'{RESULT_SAVE_PATH}/dataframe.csv', mode='a', index=False, header=True)
dataframe_train.to_csv(f'{RESULT_SAVE_PATH}/dataframe_train.csv', mode='a', index=False, header=True)
dataframe_test.to_csv(f'{RESULT_SAVE_PATH}/dataframe_test.csv', mode='a', index=False, header=True)
generated_text_df.to_csv(f'{RESULT_SAVE_PATH}/generated_text_df.csv', mode='a', index=False, header=True)
tokens_df.to_csv(f'{RESULT_SAVE_PATH}/tokens_df.csv', mode='a', index=False, header=True)
eval_loss_df.to_csv(f'{RESULT_SAVE_PATH}/eval_loss.csv', mode='a', index=False, header=True)

result = []
loss_comparison = []
for i in range(len(tokens_df)):
    x =tokens_df['original'][i]
    loss_x = eval_loss_df['original'][i]
    loss_y = np.mean([eval_loss_df['tr_1'][i], eval_loss_df['tr_2'][i], eval_loss_df['tr_3'][i]])
    max_length_y = max(tokens_df['tr_1'][i].shape[1], tokens_df['tr_2'][i].shape[1], tokens_df['tr_3'][i].shape[1])

    tokens_df['tr_1'][i] = np.pad(tokens_df['tr_1'][i][0], (0, abs(tokens_df['tr_1'][i].shape[1] - max_length_y)))
    tokens_df['tr_2'][i] = np.pad(tokens_df['tr_2'][i][0], (0, abs(tokens_df['tr_2'][i].shape[1] - max_length_y)))
    tokens_df['tr_3'][i] = np.pad(tokens_df['tr_3'][i][0], (0, abs(tokens_df['tr_3'][i].shape[1] - max_length_y)))

    y = np.mean([tokens_df['tr_1'][i], tokens_df['tr_2'][i], tokens_df['tr_3'][i]], axis=0)
    if y.shape:
        max_length = max(x.shape[1], y.shape[0])
    else:
        max_length = max(x.shape[1], 0)
    x = np.pad(x[0], (0, abs(x.shape[1] - max_length)))
    if y.shape:
        y = np.pad(y, (0, abs(y.shape[0] - max_length)))
    else:
        y = np.pad(y, (0, abs(0 - max_length)))
    result.append(attack.similarity_comparison([x], [y], 0.38))
    loss_comparison.append(attack.loss_difference(loss_x, loss_y, -1))

printTextShadi("Cosine Similarity")
printTextShadi(result)
printTextShadi("Loss Comparison")
printTextShadi(loss_comparison)

y_true = dataframe['y_true'].tolist()
y_pred = [tup[0] for tup in result]
printTextShadi("y pred for Cosine Similarity", y_pred)
printTextShadi("y true for Cosine Similarity", y_true)

printTextShadi()
printTextShadi(eval.evaluation_metrics(y_true, y_pred))
printTextShadi()
y_pred_loss = [tup[0] for tup in loss_comparison]
printTextShadi("y pred for loss comparison", y_pred_loss)
printTextShadi("y true for loss comparison", y_true)

printTextShadi()
printTextShadi(eval.evaluation_metrics(y_true, y_pred_loss))
printTextShadi()

eval.eval_roc_curve(y_true, y_pred_loss)

tpr_at_2_fpr, tpr_at_5_fpr, tpr_at_10_fpr = eval.calculate_tpr_at_fpr(y_true, y_pred_loss)
printTextShadi(f"TPR at 2% FPR: {tpr_at_2_fpr}")
printTextShadi(f"TPR at 5% FPR: {tpr_at_5_fpr}")
printTextShadi(f"TPR at 10% FPR: {tpr_at_10_fpr}")

end_time = time.time()
printTextShadi(f"Execution time: {(end_time - start_time)/60} minutes")
printTextShadi(f"Finished Program at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}")
