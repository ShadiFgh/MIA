from transformers import GPT2Tokenizer, TFGPT2Model, GPT2LMHeadModel
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import data

def printTextShadi(*args, **kwargs):
    with open('output.txt', 'a') as f:
        for arg in args:
            print(arg)
            f.write(f"{arg}\n")
        for key, value in kwargs.items():
            print(f"{key}: {value}")
            f.write(f"{key}: {value}\n")

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

# def trg_mdl_train(target_model, dataloader, device=torch.device('cpu')): 
#     # Load the target model
#     # model = GPT2LMHeadModel.from_pretrained(target_model) 
#     model = target_model
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-5) 

#     model.train()  # Set the model to training mode
#     for epoch in range(40): 
#         epoch_loss = 0  # To track the total loss for the epoch
#         for batch in tqdm(dataloader, desc=f"Training Epoch {epoch+1}"): 
#             input_ids = batch["input_ids"].to(device) 
#             optimizer.zero_grad() 
#             outputs = model(input_ids=input_ids, labels=input_ids) 
#             loss = outputs.loss 
#             loss.backward()  # Backpropagate the gradients
#             optimizer.step() 
            
#             epoch_loss += loss.item()  # Accumulate the loss
#             tqdm.write(f"Batch loss: {loss.item():.4f}")  # Optionally printTextShadi the loss for each batch
        
#         avg_loss = epoch_loss / len(dataloader)  # Calculate the average loss for the epoch
#         tqdm.write(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")  # printTextShadi the average loss for the epoch


import torch
from tqdm import tqdm

def trg_mdl_train(target_model, dataloader, device=torch.device('cpu')):
    # Load the target model
    model = target_model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    model.train()  # Set the model to training mode
    for epoch in range(100):
        epoch_loss = 0  # To track the total loss for the epoch
        for batch in tqdm(dataloader, desc=f"Training Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss

            # Check for NaN values in the loss
            if torch.isnan(loss):
                printTextShadi("NaN loss encountered")
                continue

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimization step
            optimizer.step()

            # Accumulate the loss
            epoch_loss += loss.item()
            tqdm.write(f"Batch loss: {loss.item():.4f}")  # Optionally printTextShadi the loss for each batch

        avg_loss = epoch_loss / len(dataloader)  # Calculate the average loss for the epoch
        tqdm.write(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")  # printTextShadi the average loss for the epoch




import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def generate_text(model, dataloader, tokenizer, device=torch.device('cpu')): 
    # Set model to evaluation mode 
    model.eval()
    model.to(device)
    
    all_generated_text = [] 
    all_loss = []
    all_tokens = []
    total_loss = 0.0
    num_batches = 0

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        # Evaluate on the dataset 
        for batch in dataloader:
            print(f">>>>>>>>> {batch["input_ids"].shape}")
            # Truncate input tensor if it exceded predefined limit (input+output=1024)
            max_length = 1000 - MAX_NEW_TOKENS_TO_GENERATE
            if batch['input_ids'].shape[1] > max_length:
                batch['input_ids'] = batch['input_ids'][:, :max_length]
                print(f"Truncated the input tensor shape: {batch['input_ids'].shape}") 
            try:
                input_ids = batch["input_ids"].to(device)
            except:
                printTextShadi("Shadi: Something went wrong with input_id part")
                loss = torch.tensor(0)
                all_loss.append(loss)
                # emmpty tensor of size [1,1] with zeros
                generated_text = torch.zeros((1,1))
                all_tokens.append(generated_text)
                all_generated_text.append("")
                continue
            # Check if tensor is empty
            if input_ids.nelement() == 0:
                printTextShadi("Shadi: Empty tensor, therefore skipping it")
                loss = torch.tensor(0)
                all_loss.append(loss)
                # emmpty tensor of size [1,1] with zeros
                generated_text = torch.zeros((1,1))
                all_tokens.append(generated_text)
                all_generated_text.append("")
                continue
            try:
                # Generate text 
                generated_text = model.generate( 
                    input_ids=input_ids, 
                    # max_length=100,  # Adjust max length is input + output
                    max_new_tokens=MAX_NEW_TOKENS_TO_GENERATE, # Max output tokens
                    num_return_sequences=1,  # Number of sequences to generate per input 
                    temperature=0.7,  # Adjust temperature for randomness 
                    top_k=50,  # Adjust top_k for diversity 
                    top_p=0.95,  # Adjust top_p for diversity 
                    repetition_penalty=1.2,  # Adjust repetition penalty if needed 
                    do_sample=True,  # Enable sampling 
                    pad_token_id=tokenizer.eos_token_id,  # Specify end-of-sequence token 
                    num_beams=1,  # Set num_beams to 1 for greedy search 
                    no_repeat_ngram_size=2  # Adjust no_repeat_ngram_size to avoid repeating n-grams 
                )
            except:
                printTextShadi("Shadi: Something Went Wrong with this generation, skipping it")
                loss = torch.tensor(0)
                all_loss.append(loss)
                # emmpty tensor of size [1,1] with zeros
                generated_text = torch.zeros((1,1))
                all_tokens.append(generated_text)
                all_generated_text.append("")
                continue
            
            # Calculate loss
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            printTextShadi(f"Loss: {loss}")
            all_loss.append(loss)
            # total_loss += loss.item()
            # num_batches += 1

            # Decode generated text 
            decoded_text = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in generated_text] 
            all_generated_text.append(decoded_text) 
            all_tokens.append(generated_text)
            
            # printTextShadi generated text 
            for text in decoded_text: 
                printTextShadi(f"Generated Text:\n{text}\n") 

    # Calculate average loss
    # avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    # printTextShadi(f"Average Loss: {avg_loss}")

    return all_generated_text, all_loss, all_tokens

# Example usage:
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model = GPT2LMHeadModel.from_pretrained("gpt2")
# dataloader = ...  # Your DataLoader with the validation/test dataset
# generated_texts, loss = evaluate_model(model, dataloader, tokenizer)