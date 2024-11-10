import torch
from torch.jit import script, trace
import torch.nn.functional as F
import re
import unicodedata
from io import open
import itertools
import os
import random
# import codecs
# import math

### preprocessing
def read_in(my_path):
    with open(my_path, 'r', encoding='utf-8') as dat:
        raw_lines = dat.read()
    delims = "?", "!", ".", ":"
    regex_pattern = '|'.join(map(re.escape, delims))
    raw_lines = re.split(regex_pattern, raw_lines)
    return raw_lines

# cleaning lines in .txt file
def clean_up(line):
  line = line.strip()
  line = re.sub(r"__eou__", r"", line)
  line = unicodeToAscii(line.lower().strip())
  #line = re.sub(r"'", r"", line)
  line = re.sub(r"([.'!?])", r" \1", line)
  line = re.sub(r"[^a-zA-Z.!?]+", r" ", line)
  line = re.sub(r"\s+", r" ", line).strip()
  if '\n' in line:
    line = line.replace('\n', ' ')
  return line

# Turn a Unicode string to plain ASCII
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
# normalizing string
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    # s = re.sub(r"[']", r"", s)
    s = re.sub(r"([.'?!])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

########## Vocabulary class ##########
class Voc:
    def __init__(self):
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)
        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.addWord(word)

########## creating pairs ##########

def filterPair(p, MAX_LENGTH):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs, MAX_LENGTH):
    return [pair for pair in pairs if filterPair(pair, MAX_LENGTH)]

def readVocs(dat):
    pairs = []
    voc = Voc()
    for i in range(0, len(dat)-1):
        new_combo = [dat[i], dat[i+1]]
        pairs.append(new_combo)
    return voc, pairs

def loadPrepareData(dat, MAX_LENGTH):
    voc, pairs = readVocs(dat)
    pairs = filterPairs(pairs, MAX_LENGTH)
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs

def trimRareWords(voc, pairs, MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs
    
########## preparing data for model ##########
def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.BoolTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len

########## testing out code ##########

# MAX_LENGTH = 10
# txt_path = os.path.join(os.path.abspath("data"), "dialogues_text2.txt")
# raw_lines = read_in(txt_path)
# cleaned_lines = list(map(clean_up, raw_lines))
# voc, pairs = loadPrepareData(cleaned_lines, MAX_LENGTH)
# pairs = trimRareWords(voc, pairs, 5)
# for i in pairs[50:70]: print(i)
#print(voc.word2index)
#for i in range(10): print(voc.index2word.get(i))
#print(voc.index2word.get(205))

# Example for validation
# small_batch_size = 5
# input_variable, lengths, target_variable, mask, max_target_len = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])

# print("input_variable:", input_variable)
# print("lengths:", lengths)
# print("target_variable:", target_variable)
# print("mask:", mask)
# print("max_target_len:", max_target_len)

if __name__ == "__main__":
    print("preprocessing mod")


