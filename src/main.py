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

if len(sys.argv) > 1:
    if str(sys.argv[1]).lower().strip() == "testing":
        dataset_size = 3
    else:
        try:
            dataset_size = int(sys.argv[1])
        except:
            print("Failed to parse number of lines of dataset you want to use!")
            print("Example for testing that is 3 lines for default, Use like: python main.py testing")
            print("Example for using 981 lines, Use like: python main.py 981")
            sys.exit()
    dataframe_train, dataframe_test = data.get_dataset(num_lines = dataset_size)
else:
    dataframe_train, dataframe_test = data.get_dataset()

dataframe_train['y_true'] = 1
dataframe_test['y_true'] = 0
dataframe = pd.concat([dataframe_train, dataframe_test], ignore_index=True)

dataframe['back_tr_1'] = attack.get_back_translations(dataframe['text'], 'spa_Latn')
dataframe['back_tr_2'] = attack.get_back_translations(dataframe['back_tr_1'], 'fra_Latn')
dataframe['back_tr_3'] = attack.get_back_translations(dataframe['back_tr_2'], 'deu_Latn')


tg_model = GPT2LMHeadModel.from_pretrained("gpt2", device_map="auto", torch_dtype=torch.float16)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2", device_map="auto", torch_dtype=torch.float16)
dataset = GPT2Dataset(dataframe_train['text'])
dataloader = DataLoader(dataset, shuffle=False, batch_size=1)

target_model.trg_mdl_train(target_model=tg_model, dataloader=dataloader)

dataset = GPT2Dataset(dataframe['text'])
dataloader = DataLoader(dataset, shuffle=False)
generated_text_df = pd.DataFrame()
eval_loss_df = pd.DataFrame()
tokens_df = pd.DataFrame()
generated_text_df['original'], eval_loss_df['original'], tokens_df['original'] = target_model.generate_text(tg_model, dataloader, tokenizer)

dataset_back_tr_1 = GPT2Dataset(dataframe['back_tr_1'])
dataloader = DataLoader(dataset_back_tr_1, shuffle=False, batch_size=1)
generated_text_df['tr_1'], eval_loss_df['tr_1'], tokens_df['tr_1']  = target_model.generate_text(tg_model, dataloader, tokenizer)

dataset_back_tr_2 = GPT2Dataset(dataframe['back_tr_2'])
dataloader = DataLoader(dataset_back_tr_2, shuffle=False, batch_size=1)
generated_text_df['tr_2'], eval_loss_df['tr_2'], tokens_df['tr_2'] = target_model.generate_text(tg_model, dataloader, tokenizer)

dataset_back_tr_3 = GPT2Dataset(dataframe['back_tr_3'])
dataloader = DataLoader(dataset_back_tr_3, shuffle=False, batch_size=1)
generated_text_df['tr_3'], eval_loss_df['tr_3'], tokens_df['tr_3']  = target_model.generate_text(tg_model, dataloader, tokenizer)

# dataframe.to_csv('dataframe.csv', mode='a', index=False, header=True)
# generated_text_df.to_csv('generated_text_df.csv', mode='a', index=False, header=True)
# tokens_df.to_csv('tokens_df.csv', mode='a', index=False, header=True)
# eval_loss_df.to_csv('eval_loss.csv', mode='a', index=False, header=True)

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
    max_length = max(x.shape[1], y.shape[0])
    x = np.pad(x[0], (0, abs(x.shape[1] - max_length)))
    y = np.pad(y, (0, abs(y.shape[0] - max_length)))
    result.append(attack.similarity_comparison([x], [y], 0.4))
    loss_comparison.append(attack.loss_difference(loss_x, loss_y, 0.1))

print(result)
print(loss_comparison)

y_true = dataframe['y_true'].tolist()
y_pred = [tup[0] for tup in result]
print(y_pred)
print(y_true)

print()
print(eval.evaluation_metrics(y_true, y_pred))
print()
y_pred_loss = [tup[0] for tup in loss_comparison]
print(y_pred_loss)
print(y_true)

print()
print(eval.evaluation_metrics(y_true, y_pred_loss))
print()

eval.eval_roc_curve(y_true, y_pred)