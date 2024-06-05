from transformers import GPT2Tokenizer, TFGPT2Model, GPT2LMHeadModel
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import data

class GPT2Dataset(Dataset):
  def __init__(self, dataframe):
    self.dataframe = dataframe
    self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

  def __len__(self):
    return len(self.dataframe)

  def __getitem__(self, idx):
    text = self.dataframe.iloc[idx]
    encoding = self.tokenizer(text, return_tensors="pt")
    input_ids = encoding["input_ids"].squeeze()
    return {"input_ids": input_ids}

def trg_mdl_train(target_model, dataloader): 
    # Load the target model
    # model = GPT2LMHeadModel.from_pretrained(target_model) 
    model = target_model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5) 

    device = "cuda" if torch.cuda.is_available() else "cpu" 
    model.to(device) 

    model.train()  # Set the model to training mode
    for epoch in range(30): 
        epoch_loss = 0  # To track the total loss for the epoch
        for batch in tqdm(dataloader, desc=f"Training Epoch {epoch+1}"): 
            input_ids = batch["input_ids"].to(device) 
            optimizer.zero_grad() 
            outputs = model(input_ids=input_ids, labels=input_ids) 
            loss = outputs.loss 
            loss.backward()  # Backpropagate the gradients
            optimizer.step() 
            
            epoch_loss += loss.item()  # Accumulate the loss
            tqdm.write(f"Batch loss: {loss.item():.4f}")  # Optionally print the loss for each batch
        
        avg_loss = epoch_loss / len(dataloader)  # Calculate the average loss for the epoch
        tqdm.write(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")  # Print the average loss for the epoch


import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(model, dataloader, tokenizer): 
    # Set model to evaluation mode 
    model.eval() 
    device = "cuda" if torch.cuda.is_available() else "cpu" 
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
            input_ids = batch["input_ids"].to(device) 
            
            # Generate text 
            generated_text = model.generate( 
                input_ids=input_ids, 
                max_length=100,  # Adjust max length as needed 
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
            
            # Calculate loss
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            print(f"Loss: {loss}")
            all_loss.append(loss)
            # total_loss += loss.item()
            # num_batches += 1

            # Decode generated text 
            decoded_text = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in generated_text] 
            all_generated_text.append(decoded_text) 
            all_tokens.append(generated_text)
            
            # Print generated text 
            for text in decoded_text: 
                print(f"Generated Text:\n{text}\n") 

    # Calculate average loss
    # avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    # print(f"Average Loss: {avg_loss}")

    return all_generated_text, all_loss, all_tokens

# Example usage:
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model = GPT2LMHeadModel.from_pretrained("gpt2")
# dataloader = ...  # Your DataLoader with the validation/test dataset
# generated_texts, loss = evaluate_model(model, dataloader, tokenizer)