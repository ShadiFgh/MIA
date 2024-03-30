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


def generate_text(model, dataloader, tokenizer):
    # Set model to evaluation mode
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_generated_text = []
    # Generate text from training dataset
    for batch in dataloader:
      input_ids = batch["input_ids"].to(device)
      input_ids = input_ids.to(device)

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

      # Decode generated text
      decoded_text = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in generated_text]
      all_generated_text.extend(decoded_text)

      # Print generated text
    for text in decoded_text:
        print(f"Generated Text:\n{text}\n")

    return all_generated_text