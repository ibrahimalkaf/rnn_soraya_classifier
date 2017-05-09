# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 09:33:32 2016

@author: Bing Liu (liubing@cmu.edu)

Prepare data for multi-task RNN model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

from tensorflow.python.platform import gfile

# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _UNK]

START_VOCAB_dict = dict()
START_VOCAB_dict['with_padding'] = [_PAD, _UNK]
START_VOCAB_dict['no_padding'] = [_UNK]


# -*- coding: utf-8 -*-
# Custom Tokenizers
_ORDER_ID_TOKEN = "[order_id]"
_ORDER_URL_TOKEN = "[order_url]"
_TOD_TOKEN = "[tod]"

PAD_ID = 0

UNK_ID_dict = dict()
UNK_ID_dict['with_padding'] = 1
UNK_ID_dict['no_padding'] = 0

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d")

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_PUNCTUATIONS = re.compile("([.,!?\"'])")
_DIGIT_RE = re.compile(r"\d")
_NOT_ALPHANUMERIC_RE = re.compile(r"[^a-zA-Z0-9]+")

# Modification from this References: http://daringfireball.net/2010/07/improved_regex_for_matching_urls
_URLS_REGEX = re.compile(r"(?i)\b((?:[a-z][\w-]+://(?:[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'.,<>?]))")

_ORDER_ID_REGEX = re.compile(r"((\+62|0)8[0-9]{13,})")
_ORDER_URL_REGEX = re.compile(r"(((http|https)://)?(www\.)?salestockindonesia\.com/order/history/((\+62|0)8[0-9]{13,}))")
_ORDER_URL_SPECIFIC = "https://www.salestockindonesia.com/order/history/[order_id]"
_ORDER_URL_GENERIC = "https://www.salestockindonesia.com/order/history"

_TOD_REGEX = re.compile(r"([Ss]elamat (pagi|Pagi|siang|Siang|sore|Sore|malam|Malam))")
_PUNCTUATIONS_TRIM_REGEX = re.compile(r"((\.){2,}|(\,){2,}|(\:){2,}|(\;){2,}|(\(){2,}|(\)){2,}|(\<){2,}|(\>){2,}|(\{){2,}|(\}){2,}|(\[){2,}|(\]){2,}|(\?){2,}|(\!){2,})")
_PUNCTUATIONS_COMBO_TRIM_REGEX = re.compile(r"((\.\s){2,}|(\,\s){2,}|(\:\s){2,}|(\;\s){2,}|(\?\s){2,}|(\!\s){2,}|"
                                            r"(\(\s){2,}|(\)\s){2,}|(\<\s){2,}|(\>\s){2,}|(\{\s){2,}|(\}\s){2,}|"
                                            r"(\?\!){2,})")

_SMILEY_REGEX = re.compile(r"([:|;]\s[)|(|d|p])$")

def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []

  # sentence = sentence.lower()
  sentence = _ORDER_URL_REGEX.sub(_ORDER_URL_TOKEN, sentence)
  sentence = _ORDER_ID_REGEX.sub(_ORDER_ID_TOKEN, sentence)
  sentence = _TOD_REGEX.sub(_TOD_TOKEN, sentence)

  while _PUNCTUATIONS_TRIM_REGEX.search(sentence):
    punctuation_index = _PUNCTUATIONS_TRIM_REGEX.search(sentence).start()
    punctuation_char = sentence[punctuation_index:(punctuation_index+1)]
    sentence = _PUNCTUATIONS_TRIM_REGEX.sub(punctuation_char, sentence, 1)

  while _PUNCTUATIONS_COMBO_TRIM_REGEX.search(sentence):
    punctuation_index = _PUNCTUATIONS_COMBO_TRIM_REGEX.search(sentence).start()
    punctuation_char = sentence[punctuation_index:(punctuation_index + 2)]
    sentence = _PUNCTUATIONS_COMBO_TRIM_REGEX.sub(punctuation_char, sentence, 1)

  for space_separated_fragment in sentence.strip().split():
    if _URLS_REGEX.match(space_separated_fragment):
      words.append(space_separated_fragment)
    else:
      space_separated_fragment = space_separated_fragment.lower()
      words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  return [w for w in words if w]
  
def basic_tokenizer_1(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  return [w for w in words if w]

def naive_tokenizer(sentence):
  """Naive tokenizer: split the sentence by space into a list of tokens."""
  return sentence.split()  


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
  """Create vocabulary file (if it does not exist yet) from data file.

  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="r") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 100000 == 0:
          print("  processing line %d" % counter)
        tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
        for w in tokens:
          if not _URLS_REGEX.match(w):
            word = re.sub(_DIGIT_RE, "0", w) if normalize_digits else w
          else:
            word = w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      vocab_list = START_VOCAB_dict['with_padding'] + sorted(vocab, key=vocab.get, reverse=True)
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + "\n")


def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="r") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary, UNK_ID,
                          tokenizer=None, normalize_digits=True):
  """Convert a string to list of integers representing token-ids.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

  Args:
    sentence: a string, the sentence to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.

  Returns:
    a list of integers, the token-ids for the sentence.
  """
  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(re.sub(_DIGIT_RE, "0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True, use_padding=True):
  """Tokenize data file and turn into token-ids using given vocabulary file.

  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.

  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="r") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          if use_padding:
            UNK_ID = UNK_ID_dict['with_padding']
          else:
            UNK_ID = UNK_ID_dict['no_padding']
          token_ids = sentence_to_token_ids(line, vocab, UNK_ID, tokenizer,
                                            normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")



def create_label_vocab(vocabulary_path, data_path):
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="r") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 100000 == 0:
          print("  processing line %d" % counter)
        label = line.strip()
        vocab[label] = 1
      label_list = START_VOCAB_dict['no_padding'] + sorted(vocab)
      with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
        for k in label_list:
          vocab_file.write(k + "\n")

def prepare_multi_task_data(data_dir, in_vocab_size, out_vocab_size):
    train_path = data_dir + '/train/train'
    dev_path = data_dir + '/valid/valid'
    test_path = data_dir + '/test/test'
    
    # Create vocabularies of the appropriate sizes.
    in_vocab_path = os.path.join(data_dir, "in_vocab_%d.txt" % in_vocab_size)
    out_vocab_path = os.path.join(data_dir, "out_vocab_%d.txt" % out_vocab_size)
    label_path = os.path.join(data_dir, "label.txt")
    
    create_vocabulary(in_vocab_path, train_path + ".seq.in", in_vocab_size, tokenizer=naive_tokenizer)
    create_vocabulary(out_vocab_path, train_path + ".seq.out", out_vocab_size, tokenizer=naive_tokenizer)
    create_label_vocab(label_path, train_path + ".label")
    
    # Create token ids for the training data.
    in_seq_train_ids_path = train_path + (".ids%d.seq.in" % in_vocab_size)
    out_seq_train_ids_path = train_path + (".ids%d.seq.out" % out_vocab_size)
    label_train_ids_path = train_path + (".ids.label")

    data_to_token_ids(train_path + ".seq.in", in_seq_train_ids_path, in_vocab_path, tokenizer=naive_tokenizer)
    data_to_token_ids(train_path + ".seq.out", out_seq_train_ids_path, out_vocab_path, tokenizer=naive_tokenizer)
    data_to_token_ids(train_path + ".label", label_train_ids_path, label_path, normalize_digits=False, use_padding=False)
    
    # Create token ids for the development data.
    in_seq_dev_ids_path = dev_path + (".ids%d.seq.in" % in_vocab_size)
    out_seq_dev_ids_path = dev_path + (".ids%d.seq.out" % out_vocab_size)
    label_dev_ids_path = dev_path + (".ids.label")

    data_to_token_ids(dev_path + ".seq.in", in_seq_dev_ids_path, in_vocab_path, tokenizer=naive_tokenizer)
    data_to_token_ids(dev_path + ".seq.out", out_seq_dev_ids_path, out_vocab_path, tokenizer=naive_tokenizer)
    data_to_token_ids(dev_path + ".label", label_dev_ids_path, label_path, normalize_digits=False, use_padding=False)
    
    # Create token ids for the test data.
    in_seq_test_ids_path = test_path + (".ids%d.seq.in" % in_vocab_size)
    out_seq_test_ids_path = test_path + (".ids%d.seq.out" % out_vocab_size)
    label_test_ids_path = test_path + (".ids.label")
    
    data_to_token_ids(test_path + ".seq.in", in_seq_test_ids_path, in_vocab_path, tokenizer=naive_tokenizer)
    data_to_token_ids(test_path + ".seq.out", out_seq_test_ids_path, out_vocab_path, tokenizer=naive_tokenizer)
    data_to_token_ids(test_path + ".label", label_test_ids_path, label_path, normalize_digits=False, use_padding=False)
    
    return (in_seq_train_ids_path, out_seq_train_ids_path, label_train_ids_path,
          in_seq_dev_ids_path, out_seq_dev_ids_path, label_dev_ids_path,
          in_seq_test_ids_path, out_seq_test_ids_path, label_test_ids_path,
          in_vocab_path, out_vocab_path, label_path)