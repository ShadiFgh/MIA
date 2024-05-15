import attack
import data
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import DataLoader
from target_model import GPT2Dataset
import target_model

# text = data.get_dataset()
# print(text)
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
dataframe = data.get_dataset()
dataset = GPT2Dataset(dataframe['text'])
dataloader = DataLoader(dataset, shuffle=False)

back_tr_text = attack.get_back_translations(dataframe, 'spa_Latn')
print(back_tr_text)

target_model.trg_mdl_train(target_model=model, dataloader=dataloader)