import os
import re
import preprocessing as prep_mod

model_name = 'chat_model'
model_path = os.path.join(os.path.abspath('model'), model_name)
txt_path = os.path.join(os.path.abspath('data'), 'dialogues_text2.txt')

def read_in(my_path):
    with open(my_path, 'r', encoding='utf8') as dat:
        raw_lines = dat.read()
    delims = "?", "!", ".", ":"
    regex_pattern = '|'.join(map(re.escape, delims))
    raw_lines = re.split(regex_pattern, raw_lines)
    return raw_lines

def clean_up(line):
  line = line.strip()
  line = re.sub(r"__eou__", r"", line)
  #line = re.sub(r"'", r"", line)
  if '\n' in line:
    line = line.replace('\n', ' ')
  return line

raw_lines = read_in(txt_path)
#print(raw_lines[:10])

cleaned_lines = list(map(clean_up, raw_lines))
print(cleaned_lines[:20])

# hidden_size = 200
# encoder_n_layers = 2
# decoder_n_layers = 2
# dropout = 0.1
# batch_size = 1
# checkpoint_iter = 3000
# loadFilename = os.path.join(model_path,
#                 '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
#                 '{}_checkpoint.tar'.format(checkpoint_iter))
# mypath = os.path.join(os.path.abspath('data'), 'dialogues_text2.txt')

# print(loadFilename)
# print(os.path.exists(loadFilename))
#print(txt_path)