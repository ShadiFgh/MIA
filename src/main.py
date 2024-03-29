import attack
import data
# from target_model import GPT2Dataset
# import target_model

text = data.get_dataset()
print(text)

back_tr_text = attack.get_back_translations(text)
print(back_tr_text)