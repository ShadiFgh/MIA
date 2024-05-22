import numpy as np
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics.pairwise import cosine_similarity



def generate_back_translations(text, tgt_language):

  checkpoint = 'facebook/nllb-200-distilled-600M'
  model=AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
  tokenizer=AutoTokenizer.from_pretrained(checkpoint)

  translator = pipeline("translation", model=model, tokenizer=tokenizer, src_lang='eng_Latn', tgt_lang=tgt_language)
  back_translator = pipeline("translation", model=model, tokenizer=tokenizer, src_lang=tgt_language, tgt_lang='eng_Latn')

  translated_text = translator(text)
  back_translated_text = back_translator(translated_text[0]['translation_text'])

  return back_translated_text[0]['translation_text']


def get_back_translations(dataset, target_lan):

  return dataset.apply(generate_back_translations, tgt_language=target_lan)


def similarity_comparison(x, y, w):

  cosine = cosine_similarity(x, y)
  if cosine < w:
    return 'in', cosine
  elif cosine > w:
    return 'out', cosine
  

def loss_difference(x, y, w):

  difference = abs(x - y)
  if difference < w:
    return 'out', difference
  elif difference > w:
    return 'in', difference