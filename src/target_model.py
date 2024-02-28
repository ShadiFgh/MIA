from transformers import GPT2Tokenizer, TFGPT2Model, GPT2LMHeadModel
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class GPT2Dataset(Dataset):
  def __init__(self, dataframe):
    self.dataframe = dataframe
    self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

  def __len__(self):
    return len(self.dataframe)

  def __getitem__(self, idx):
    text = self.dataframe.iloc[idx]["text"]
    encoding = self.tokenizer(text, return_tensors="pt")
    input_ids = encoding["input_ids"].squeeze()
    return {"input_ids": input_ids}

dataset = GPT2Dataset(dataset1)
dataloader = DataLoader(dataset, shuffle=True)

model = GPT2LMHeadModel.from_pretrained("gpt2")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

for epoch in range(3):
  for batch in tqdm(dataloader):
    input_ids = batch["input_ids"].to(device)
    outputs = model(input_ids=input_ids, labels=input_ids)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()