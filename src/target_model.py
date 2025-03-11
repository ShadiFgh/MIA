from transformers import GPT2Tokenizer, TFGPT2Model, GPT2LMHeadModel
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import data
import printtextme
from printtextme import printTextme
from private_transformers import PrivacyEngine
import torch.nn.functional as F

MAX_NEW_TOKENS_TO_GENERATE = 200

class GPT2Dataset(Dataset):
  def __init__(self, dataframe):
    self.dataframe = dataframe
    self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

  def __len__(self):
    return len(self.dataframe)

  def __getitem__(self, idx):
    text = self.dataframe.iloc[idx]
    if isinstance(text, float):
        text = ""
    encoding = self.tokenizer(text, return_tensors="pt")
    input_ids = encoding["input_ids"].squeeze()
    return {"input_ids": input_ids}



import torch
from tqdm import tqdm

def trg_mdl_train(target_model, dataloader, device=torch.device('cpu')):
    # Load the target model
    model = target_model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    privacy_engine = PrivacyEngine(
    model,
    batch_size=1,
    sample_size=50000,
    epochs=100,
    max_grad_norm=0.1,
    target_epsilon=1,
)
    privacy_engine.attach(optimizer)

    model.train()  # Set the model to training mode

    max_length = 1024  # Maximum length for GPT-2

    for epoch in range(100):
        epoch_loss = 0  # To track the total loss for the epoch
        for batch in tqdm(dataloader, desc=f"Training Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)

            # Truncate input sequences longer than the max_length
            if input_ids.shape[1] > max_length:
                input_ids = input_ids[:, :max_length]  # Truncate

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids=input_ids, labels=input_ids)
            # loss = outputs.loss
            labels = input_ids[:, 1:, ]
            logits = outputs.logits[:, :-1, :].permute(0, 2, 1)
            # `loss` is a 1-D tensor of shape (batch_size,).
            loss = F.cross_entropy(logits, labels, reduction="none").mean(dim=1)

            # Check for NaN values in the loss
            if torch.isnan(loss):
                printTextme("NaN loss encountered")
                continue

            # Backward pass
            # loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimization step
            optimizer.step(loss=loss)

            # Accumulate the loss
            epoch_loss += loss.item()
            tqdm.write(f"Batch loss: {loss.item():.4f}")  # Optionally printTextme the loss for each batch

        avg_loss = epoch_loss / len(dataloader)  # Calculate the average loss for the epoch
        tqdm.write(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")





import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"




# THIS FUNCTION IS USED TO GENERATE LOSS FROM THE MODEL (Will rename it later)
def generate_text(model, dataloader, tokenizer, device=torch.device('cpu')): 
    # Set model to evaluation mode 
    model.eval()
    model.to(device)
    
    all_loss = []
    total_loss = 0.0
    num_batches = 0

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        # Evaluate on the dataset 
        for batch in dataloader:
            try:
                # Handle input size and truncation if necessary
                max_length = 1000 - MAX_NEW_TOKENS_TO_GENERATE
                if batch['input_ids'].shape[1] > max_length:
                    batch['input_ids'] = batch['input_ids'][:, :max_length]
                
                input_ids = batch["input_ids"].to(device)
            except:
                printTextme("Something went wrong with input_id part")
                loss = torch.tensor(0)
                all_loss.append(loss)
                continue
            
            # Check if tensor is empty
            if input_ids.nelement() == 0:
                printTextme("Empty tensor, skipping it")
                loss = torch.tensor(0)
                all_loss.append(loss)
                continue
            
            try:
                # Forward pass to calculate the loss (without generating text)
                outputs = model(input_ids=input_ids, labels=input_ids)
                loss = outputs.loss
                printTextme(f"Loss: {loss.item()}")
                all_loss.append(loss.item())
                
                total_loss += loss.item()
                num_batches += 1
            except:
                printTextme("Something went wrong during loss calculation, skipping it")
                loss = torch.tensor(0)
                all_loss.append(loss.item())
                continue
    
    return all_loss
