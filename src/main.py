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
import printtextShadi
from printtextShadi import printTextShadi
import argparse
import io

def secs_to_hrs_min_secs_str(secs):
    secs = float(secs)
    # Calculate hours
    hours = secs // 3600
    # Calculate remaining seconds after hours
    remaining_secs = secs % 3600
    # Calculate minutes
    minutes = remaining_secs // 60
    # Calculate remaining seconds after minutes
    seconds = remaining_secs % 60
    # Return formatted string
    return f"{hours} hours, {minutes} minutes, {seconds} seconds"

device_type = 'cuda' # cuda or cpu
device_id = 2
DATASET_BATCH_SIZE = None
DATASET_NAME = None
SELECTED_BATCH = None
LOAD_TRAIN_TEST_DATA_FRAME = False
LOAD_BACK_TRANSLATIONS_DATA_FRAME = False
LOAD_MODEL = False
EXIT_ON_BACKTRANSLATION_COMPLETE = False
RESULT_SAVE_PATH = "DB"
dataframe_test, dataframe_train = None, None


# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Process and manage datasets with specific configurations.')
    
# Add arguments with long and short options
parser.add_argument('--device_type', '-d', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device type: "cpu" or "cuda".')
parser.add_argument('--device_id', '-i', type=int, default=0, help='Device ID to use if "cuda" is selected.')
parser.add_argument('--batch_size', '-b', type=int, default=None, help='Dataset batch size.')
parser.add_argument('--dataset_name', '-n', type=str, default=None, help='Name of the dataset.')
parser.add_argument('--selected_batch', '-s', type=int, default=None, help='Selected batch number.')
parser.add_argument('--load_train_test_data_frame', '-lttdf', type=bool, default=False, help='Flag to load test train data frame.')
parser.add_argument('--load_back_translatio_data_frame', '-lbtdf', type=bool, default=False, help='Flag to load backtrainslations data frame.')
parser.add_argument('--load_model', '-lm', type=bool, default=False, help='Flag to load model.')
parser.add_argument('--result_save_path', '-rsp', type=str, default="Result", help='Path to save the result.')
parser.add_argument('--exit_on_backtranslation_complete', '-ebc', type=bool, default=False, help='Flag to exit when backtranslation is complete.')

# Parse the arguments
args = parser.parse_args()

# Access the arguments
device_type = args.device_type
device_id = args.device_id
DATASET_BATCH_SIZE = args.batch_size
DATASET_NAME = args.dataset_name
SELECTED_BATCH = args.selected_batch
LOAD_TRAIN_TEST_DATA_FRAME = args.load_train_test_data_frame
LOAD_BACK_TRANSLATIONS_DATA_FRAME = args.load_back_translatio_data_frame
LOAD_MODEL = args.load_model
RESULT_SAVE_PATH = args.result_save_path
EXIT_ON_BACKTRANSLATION_COMPLETE = args.exit_on_backtranslation_complete

# Get the current datetime
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

# If Data is not loaded thenadd current datetime to make folder names unique
if not LOAD_BACK_TRANSLATIONS_DATA_FRAME and not LOAD_MODEL and not LOAD_TRAIN_TEST_DATA_FRAME:
    # Add current_datetime, DATASET_NAME and DATASET_BATCH_SIZE to RESULT_SAVE_PATH such that folder name is descriptive and unique
    dataset_name_string = '' if not DATASET_NAME else DATASET_NAME
    dataset_batch_size_string = 'FullData' if not DATASET_BATCH_SIZE else DATASET_BATCH_SIZE
    selected_batch_string = '' if SELECTED_BATCH == None else SELECTED_BATCH
    RESULT_SAVE_PATH = f"{RESULT_SAVE_PATH}_{current_datetime}_{dataset_name_string}_{dataset_batch_size_string}_{selected_batch_string}"


# Check if the path exists, and create it if it doesn't
if not os.path.exists(RESULT_SAVE_PATH):
    os.makedirs(RESULT_SAVE_PATH)

# Set Env variable RESULT_SAVE_PATH
os.environ['RESULT_SAVE_PATH'] = RESULT_SAVE_PATH

# Empty Output File
with open(f'{RESULT_SAVE_PATH}/output.txt', 'w') as f:
    f.write('')

start_time = time.time()
printTextShadi(f'Started Program at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
printTextShadi(f"Using the Following configuration:\n1. Device Type: {device_type}\n2. Device ID: {device_id}\n3. Dataset Batch Size: {DATASET_BATCH_SIZE}\n4. Dataset Name: {DATASET_NAME}\n5. Selected Batch: {SELECTED_BATCH}\n6. Load Train/Test Data Frame: {LOAD_TRAIN_TEST_DATA_FRAME}\n7. Load Back Translations Data Frame: {LOAD_BACK_TRANSLATIONS_DATA_FRAME}\n8. Load Model: {LOAD_MODEL}\n9. Result Save Path: {RESULT_SAVE_PATH}\n10. Exit on Backtranslation Complete: {EXIT_ON_BACKTRANSLATION_COMPLETE}")

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

# Custom Unpickler to load the model
class custom_unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        global device
        if module=='torch.storage' and name=='_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=device)
        else:
            return super().find_class(module, name)

# Get Dataset
if not LOAD_TRAIN_TEST_DATA_FRAME:
    dataframe_train, dataframe_test = data.get_dataset(DATASET_BATCH_SIZE, DATASET_NAME, SELECTED_BATCH)
    dataframe_train['y_true'] = 1
    dataframe_test['y_true'] = 0
else:
    dataframe_train = pd.read_csv(f'{RESULT_SAVE_PATH}/dataframe_train.csv')
    printTextShadi(f"Loaded Train Dataframe with Labels from {RESULT_SAVE_PATH}/dataframe_train.csv")
    dataframe_test = pd.read_csv(f'{RESULT_SAVE_PATH}/dataframe_test.csv')
    printTextShadi(f"Loaded Test Dataframe with Labels from {RESULT_SAVE_PATH}/dataframe_train.csv")
    dataframe_train['y_true'] = 1
    dataframe_test['y_true'] = 0
if not LOAD_TRAIN_TEST_DATA_FRAME:
    # Save train and test Dataframes
    dataframe_train.to_csv(f'{RESULT_SAVE_PATH}/dataframe_train.csv', index=False, header=True)
    printTextShadi("Saved Train Dataframe with Labels to dataframe_train.csv")
    dataframe_test.to_csv(f'{RESULT_SAVE_PATH}/dataframe_test.csv', index=False, header=True)
    printTextShadi("Saved Test Dataframe with Labels to dataframe_test.csv")

# Concat train and test Dataframes for calculating their backtranslations
dataframe = pd.concat([dataframe_train, dataframe_test], ignore_index=True)
# deep copy of original dataframe
original_df_copy = dataframe.copy()

log_lines = []
if LOAD_BACK_TRANSLATIONS_DATA_FRAME and os.path.exists(f'{RESULT_SAVE_PATH}/dataframe_backtr1.csv') :
    if not os.path.exists(f'{RESULT_SAVE_PATH}/dataframe_backtr2.csv') and not os.path.exists(f'{RESULT_SAVE_PATH}/dataframe_backtr3.csv') and not os.path.exists(f'{RESULT_SAVE_PATH}/dataframe_backtr4.csv') and not os.path.exists(f'{RESULT_SAVE_PATH}/dataframe_backtr5.csv'):
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

if LOAD_BACK_TRANSLATIONS_DATA_FRAME and os.path.exists(f'{RESULT_SAVE_PATH}/dataframe_backtr2.csv'):
    if not os.path.exists(f'{RESULT_SAVE_PATH}/dataframe_backtr3.csv') and not os.path.exists(f'{RESULT_SAVE_PATH}/dataframe_backtr4.csv') and not os.path.exists(f'{RESULT_SAVE_PATH}/dataframe_backtr5.csv'):
        dataframe = pd.read_csv(f'{RESULT_SAVE_PATH}/dataframe_backtr2.csv')
        log_lines.append("Loaded Backtranslation 2 from dataframe_backtr2.csv")
        printTextShadi("Loaded Backtranslation 2 from dataframe_backtr2.csv")
else:
    dataframe['back_tr_2'] = attack.get_back_translations(dataframe['text'], 'fra_Latn', device=device)
    log_lines.append("Completed Backtranslation 2")
    printTextShadi("Completed Backtranslation 2")
    # Save dataframe for backtr2
    dataframe.to_csv(f'{RESULT_SAVE_PATH}/dataframe_backtr2.csv', index=False, header=True)
    log_lines.append("Saved Backtranslation 2 to dataframe_backtr2.csv")
    printTextShadi("Saved Backtranslation 2 to dataframe_backtr2.csv")

if LOAD_BACK_TRANSLATIONS_DATA_FRAME and os.path.exists(f'{RESULT_SAVE_PATH}/dataframe_backtr3.csv'):
    if not os.path.exists(f'{RESULT_SAVE_PATH}/dataframe_backtr4.csv') and not os.path.exists(f'{RESULT_SAVE_PATH}/dataframe_backtr5.csv'):
        dataframe = pd.read_csv(f'{RESULT_SAVE_PATH}/dataframe_backtr3.csv')
        log_lines.append("Loaded Backtranslation 3 from dataframe_backtr3.csv")
        printTextShadi("Loaded Backtranslation 3 from dataframe_backtr3.csv")
else:
    dataframe['back_tr_3'] = attack.get_back_translations(dataframe['text'], 'deu_Latn', device=device)
    log_lines.append("Completed Backtranslation 3")
    printTextShadi("Completed Backtranslation 3")
    # Save dataframe for backtr3
    dataframe.to_csv(f'{RESULT_SAVE_PATH}/dataframe_backtr3.csv', index=False, header=True)
    log_lines.append("Saved Backtranslation 3 to dataframe_backtr3.csv")
    printTextShadi("Saved Backtranslation 3 to dataframe_backtr3.csv")

if LOAD_BACK_TRANSLATIONS_DATA_FRAME and os.path.exists(f'{RESULT_SAVE_PATH}/dataframe_backtr4.csv'):
    if not os.path.exists(f'{RESULT_SAVE_PATH}/dataframe_backtr5.csv'):
        dataframe = pd.read_csv(f'{RESULT_SAVE_PATH}/dataframe_backtr4.csv')
        log_lines.append("Loaded Backtranslation 4 from dataframe_backtr4.csv")
        printTextShadi("Loaded Backtranslation 4 from dataframe_backtr4.csv")
else:
    dataframe['back_tr_4'] = attack.get_back_translations(dataframe['text'], 'pes_Arab', device=device)
    log_lines.append("Completed Backtranslation 4")
    printTextShadi("Completed Backtranslation 4")
    # Save dataframe for backtr4
    dataframe.to_csv(f'{RESULT_SAVE_PATH}/dataframe_backtr4.csv', index=False, header=True)
    log_lines.append("Saved Backtranslation 4 to dataframe_backtr4.csv")
    printTextShadi("Saved Backtranslation 4 to dataframe_backtr4.csv")

if LOAD_BACK_TRANSLATIONS_DATA_FRAME and os.path.exists(f'{RESULT_SAVE_PATH}/dataframe_backtr5.csv'):
    dataframe = pd.read_csv(f'{RESULT_SAVE_PATH}/dataframe_backtr5.csv')
    log_lines.append("Loaded Backtranslation 5 from dataframe_backtr5.csv")
    printTextShadi("Loaded Backtranslation 5 from dataframe_backtr5.csv")
else:
    dataframe['back_tr_5'] = attack.get_back_translations(dataframe['text'], 'zho_Hans', device=device)
    log_lines.append("Completed Backtranslation 5")
    printTextShadi("Completed Backtranslation 5")
    # Save dataframe for backtr5
    dataframe.to_csv(f'{RESULT_SAVE_PATH}/dataframe_backtr5.csv', index=False, header=True)
    log_lines.append("Saved Backtranslation 5 to dataframe_backtr5.csv")
    printTextShadi("Saved Backtranslation 5 to dataframe_backtr5.csv")

# if dataframe is not having any row then raise an error
if dataframe.shape[0] == 0:
    raise ValueError("Dataframe is empty")
# If all lines of dataframe are same as original dataframe then raise an error
# if dataframe.equals(original_df_copy):# and 1==2:
#     raise ValueError("Dataframe is same as original dataframe")

back_translation_end_time = time.time()
time_taken_till_back_translations = secs_to_hrs_min_secs_str(back_translation_end_time - start_time)
printTextShadi(f"Time taken for back translation to complete: {time_taken_till_back_translations}")

if EXIT_ON_BACKTRANSLATION_COMPLETE:
    printTextShadi("Option EXIT_ON_BACKTRANSLATION_COMPLETE was set to True. Therefore Exiting.")
    sys.exit()
    
printTextShadi("Starting training phase...")
training_start_time = time.time()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2", torch_dtype=torch.float32)
tg_model = None
if LOAD_MODEL and os.path.exists(f'{RESULT_SAVE_PATH}/trained_model.pkl'):
    # Initialize the model
    tg_model = GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=torch.float32)
    # Load the state dictionary
    with open(f'{RESULT_SAVE_PATH}/trained_model.pkl', 'rb') as f:
        # state_dict = pickle.load(f)
        state_dict = custom_unpickler(f).load()
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

training_end_time = time.time()
time_taken_for_training = secs_to_hrs_min_secs_str(training_end_time - training_start_time)
printTextShadi(f"Time taken for training to complete: {time_taken_for_training}")

# tg_model = GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=torch.float16)
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2", torch_dtype=torch.float16)

text_generation_start_time = time.time()
printTextShadi("Get Loss values phase...")
dataset = GPT2Dataset(dataframe['text'])
dataloader = DataLoader(dataset, shuffle=False)
generated_text_df = pd.DataFrame()
eval_loss_df = pd.DataFrame()
tokens_df = pd.DataFrame()
logits_df = pd.DataFrame()

# loggits = target_model.get_scores_from_logits(tg_model, dataloader, tokenizer, device=device)
# loggits.to_csv(f'{RESULT_SAVE_PATH}/logits.csv', mode='a', index=False, header=True)
# printTextShadi("Saved the logits to logits_df.csv")

resp = target_model.generate_text(tg_model, dataloader, tokenizer, device=device)
# generated_text_df['original'] = resp[0]
eval_loss_df['original'] = resp
# tokens_df['original'] = [ten.cpu().numpy() for ten in resp[2]]

dataset_back_tr_1 = GPT2Dataset(dataframe['back_tr_1'])
dataloader = DataLoader(dataset_back_tr_1, shuffle=False, batch_size=1)
resp2= target_model.generate_text(tg_model, dataloader, tokenizer, device=device)
# generated_text_df['tr_1'] = resp2[0]
eval_loss_df['tr_1'] = resp2
# tokens_df['tr_1'] = [ten.cpu().numpy() for ten in resp2[2]]

dataset_back_tr_2 = GPT2Dataset(dataframe['back_tr_2'])
dataloader = DataLoader(dataset_back_tr_2, shuffle=False, batch_size=1)
resp3= target_model.generate_text(tg_model, dataloader, tokenizer, device=device)
# generated_text_df['tr_2'] = resp3[0]
eval_loss_df['tr_2'] = resp3
# tokens_df['tr_2'] = [ten.cpu().numpy() for ten in resp3[2]]

dataset_back_tr_3 = GPT2Dataset(dataframe['back_tr_3'])
dataloader = DataLoader(dataset_back_tr_3, shuffle=False, batch_size=1)
resp4= target_model.generate_text(tg_model, dataloader, tokenizer, device=device)
# generated_text_df['tr_3'] = resp4[0]
eval_loss_df['tr_3'] = resp4
# tokens_df['tr_3'] = [ten.cpu().numpy() for ten in resp4[2]]

dataset_back_tr_4 = GPT2Dataset(dataframe['back_tr_4'])
dataloader = DataLoader(dataset_back_tr_4, shuffle=False, batch_size=1)
resp5= target_model.generate_text(tg_model, dataloader, tokenizer, device=device)
# generated_text_df['tr_4'] = resp5[0]
eval_loss_df['tr_4'] = resp5
# tokens_df['tr_4'] = [ten.cpu().numpy() for ten in resp5[2]]

dataset_back_tr_5 = GPT2Dataset(dataframe['back_tr_5'])
dataloader = DataLoader(dataset_back_tr_5, shuffle=False, batch_size=1)
resp6= target_model.generate_text(tg_model, dataloader, tokenizer, device=device)
# generated_text_df['tr_5'] = resp6[0]
eval_loss_df['tr_5'] = resp6
# tokens_df['tr_5'] = [ten.cpu().numpy() for ten in resp6[2]]

# generated_text_df.to_csv(f'{RESULT_SAVE_PATH}/generated_text_df.csv', mode='a', index=False, header=True)
# printTextShadi("Saved the generated texts to generated_text_df.csv")
# tokens_df.to_csv(f'{RESULT_SAVE_PATH}/tokens_df.csv', mode='a', index=False, header=True)
# printTextShadi("Saved the tokens to tokens_df.csv")
eval_loss_df.to_csv(f'{RESULT_SAVE_PATH}/eval_loss.csv', mode='a', index=False, header=True)
printTextShadi("Saved the loss values to eval_loss.csv")

text_generation_end_time = time.time()
time_taken_for_text_generation = secs_to_hrs_min_secs_str(text_generation_end_time - text_generation_start_time)
printTextShadi(f"Time taken for text generation to complete: {time_taken_for_text_generation}")

# result = []
loss_comparison = []
for i in range(len(loss_comparison)):
    # x =tokens_df['original'][i]
    loss_x = eval_loss_df['original'][i]
    loss_y = np.mean([eval_loss_df['tr_1'][i], eval_loss_df['tr_2'][i], eval_loss_df['tr_3'][i]])

    loss_comparison.append(attack.loss_difference(loss_x, loss_y, -1))


printTextShadi("Loss Comparison")
printTextShadi(loss_comparison)

y_true = dataframe['y_true'].tolist()


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
