import numpy as np
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics.pairwise import cosine_similarity
import torch


def generate_back_translations(text, tgt_language, device=torch.device('cpu')):

  checkpoint = 'facebook/nllb-200-distilled-600M'
  model=AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
  model.to(device)
  tokenizer=AutoTokenizer.from_pretrained(checkpoint)

  translator = pipeline("translation", model=model, tokenizer=tokenizer, src_lang='eng_Latn', tgt_lang=tgt_language, max_length=400)
  back_translator = pipeline("translation", model=model, tokenizer=tokenizer, src_lang=tgt_language, tgt_lang='eng_Latn', max_length=400)

  translated_text = translator(text)
  back_translated_text = back_translator(translated_text[0]['translation_text'])

  return back_translated_text[0]['translation_text']


def get_back_translations(dataset, target_lan, device=torch.device('cpu')):
  return dataset.apply(generate_back_translations, tgt_language=target_lan, device=device)


def similarity_comparison(x, y, w):

  cosine = cosine_similarity(x, y)
  if cosine < w:
    return 1, cosine
  elif cosine > w:
    return 0, cosine
  

def loss_difference(x, y, w):

  difference = x - y
  if difference < w:
    return 1, difference
  elif difference > w:
    return 0, difference