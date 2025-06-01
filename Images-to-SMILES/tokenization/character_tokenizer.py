import os
import json
import numpy as np
import tensorflow as tf

class CharacterTokenizer:
    def __init__(self, mapping_json_path, special_tokens=None):
        self.mapping_path = mapping_json_path
        self.special_tokens = special_tokens if special_tokens else ['<pad>', '<start>', '<end>']
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        self.smiles_list = []

        self._load_smiles()
        self._build_vocab()

    def _load_smiles(self):
        with open(self.mapping_path, 'r') as f:
            data = json.load(f)
        self.smiles_list = [entry['smiles'] for entry in data]
    
    def _generate_statistics(self):
        lengths = [len(smile) for smile in self.smiles_list]
        self.max_allowed_length = int(np.percentile(lengths, 95))

    def _build_vocab(self):
        char_set = set()
        for smile in self.smiles_list:
            char_set.update(list(smile))

        char_list = sorted(list(char_set)) + self.special_tokens
        self.char_to_idx = {char: idx for idx, char in enumerate(char_list)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(char_list)

        print(f"Vocabulary Size: {self.vocab_size}")
        print(f"Character to Index mapping:\n{self.char_to_idx}")

    def encode(self, smile, max_len = None):
        if max_len is None:
            max_len = self.max_allowed_length
        tokens = ['<start>'] + list(smile) + ['<end>']
        token_ids = [self.char_to_idx[char] for char in tokens]

        # Pad or truncate
        if len(token_ids) < max_len:
            token_ids += [self.char_to_idx['<pad>']] * (max_len - len(token_ids))
        else:
            token_ids = token_ids[:max_len]

        return np.array(token_ids)

    def decode(self, token_ids):
        chars = [self.idx_to_char[idx] for idx in token_ids if self.idx_to_char[idx] != '<pad>']
        return ''.join(chars)