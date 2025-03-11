import numpy as np
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics.pairwise import cosine_similarity
import torch
import printtextme
from printtextme import printTextme

# def generate_back_translations(text, tgt_language, device=torch.device('cpu')):

#   checkpoint = 'facebook/nllb-200-distilled-600M'
#   model=AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
#   model.to(device)
#   tokenizer=AutoTokenizer.from_pretrained(checkpoint)

#   translator = pipeline("translation", model=model, tokenizer=tokenizer, src_lang='eng_Latn', tgt_lang=tgt_language, max_length=400, truncation=True)
#   back_translator = pipeline("translation", model=model, tokenizer=tokenizer, src_lang=tgt_language, tgt_lang='eng_Latn', max_length=400, truncation=True)

#   if len(text) > 400:
#     print(f"Long text{len(text)}")
#   translated_text = translator(text)
#   back_translated_text = back_translator(translated_text[0]['translation_text'])

#   return back_translated_text[0]['translation_text']


# def get_back_translations(dataset, target_lan, device=torch.device('cpu')):
#   return dataset.apply(generate_back_translations, tgt_language=target_lan, device=device)


def generate_back_translations(text, tgt_language_1, tgt_language_2, device=torch.device('cpu')):
  
    checkpoint = 'facebook/nllb-200-distilled-600M'
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # Pipelines for translation steps
    translator_1 = pipeline("translation", model=model, tokenizer=tokenizer, src_lang='eng_Latn', tgt_lang=tgt_language_1, max_length=400, truncation=True)
    translator_2 = pipeline("translation", model=model, tokenizer=tokenizer, src_lang=tgt_language_1, tgt_lang=tgt_language_2, max_length=400, truncation=True)
    back_translator = pipeline("translation", model=model, tokenizer=tokenizer, src_lang=tgt_language_2, tgt_lang='eng_Latn', max_length=400, truncation=True)

    if len(text) > 400:
        print(f"Long text {len(text)}")
    
    # Step-by-step translation
    translated_text_1 = translator_1(text)
    translated_text_2 = translator_2(translated_text_1[0]['translation_text'])
    back_translated_text = back_translator(translated_text_2[0]['translation_text'])

    return back_translated_text[0]['translation_text']

def get_back_translations(dataset, target_lan_1, target_lan_2, device=torch.device('cpu')):
    return dataset.apply(generate_back_translations, tgt_language_1=target_lan_1, tgt_language_2=target_lan_2, device=device)



  

def loss_difference(x, y, w):

  difference = x - y
  if difference < w:
    return 1, difference
  elif difference > w:
    return 0, difference
