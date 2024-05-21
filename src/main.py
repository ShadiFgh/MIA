import attack
import data
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import DataLoader
from target_model import GPT2Dataset
import target_model
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


dataframe = data.get_dataset()
dataframe['back_tr_1'] = attack.get_back_translations(dataframe['text'], 'spa_Latn')
dataframe['back_tr_2'] = attack.get_back_translations(dataframe['back_tr_1'], 'fra_Latn')
dataframe['back_tr_3'] = attack.get_back_translations(dataframe['back_tr_2'], 'deu_Latn')


tg_model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
dataset = GPT2Dataset(dataframe['text'][0:3])
dataloader = DataLoader(dataset, shuffle=False)

target_model.trg_mdl_train(target_model=tg_model, dataloader=dataloader)

dataset = GPT2Dataset(dataframe['text'])
dataloader = DataLoader(dataset, shuffle=False)
generated_text_df = pd.DataFrame()
generated_text_df['original'] = target_model.generate_text(tg_model, dataloader, tokenizer)

dataset_back_tr_1 = GPT2Dataset(dataframe['back_tr_1'])
dataloader = DataLoader(dataset_back_tr_1, shuffle=False)
generated_text_df['tr_1'] = target_model.generate_text(tg_model, dataloader, tokenizer)

dataset_back_tr_2 = GPT2Dataset(dataframe['back_tr_2'])
dataloader = DataLoader(dataset_back_tr_2, shuffle=False)
generated_text_df['tr_2'] = target_model.generate_text(tg_model, dataloader, tokenizer)

dataset_back_tr_3 = GPT2Dataset(dataframe['back_tr_3'])
dataloader = DataLoader(dataset_back_tr_3, shuffle=False)
generated_text_df['tr_3'] = target_model.generate_text(tg_model, dataloader, tokenizer)

vectorizer = TfidfVectorizer()
for i in range(len(generated_text_df)):
    x = vectorizer.fit_transform([generated_text_df['original'][i]])
    y_corpus = []
    for col in ['tr_1', 'tr_2', 'tr_3']:
        y_corpus.append(generated_text_df.loc[i, col])
    y = np.mean(vectorizer.transform(y_corpus).toarray(), axis=0)
    print(attack.similarity_comparison(x, y.reshape(1, -1), 0.1))