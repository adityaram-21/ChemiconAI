import json
import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

class SmilesToIUPACPreprocessor:
    def __init__(self, raw_csv_path, smiles_vocab_path="smiles_vocab.json", iupac_vocab_path="iupac_vocab.json"):
        self.raw_csv_path = raw_csv_path
        self.smiles_vocab_path = smiles_vocab_path
        self.iupac_vocab_path = iupac_vocab_path
        self.df = None
        self.char2idx_smiles = {}
        self.idx2char_smiles = {}
        self.char2idx_iupac = {}
        self.idx2char_iupac = {}
        self.smiles_tensors = []
        self.iupac_tensors = []
        self.padded_smiles = None
        self.padded_iupac = None
    
    def preprocess_data(self):
        """
        Preprocess the input CSV file to clean and filter the data.
        """
        self.csv_path = "data/smiles_to_iupac_clean.csv"
        # Read the input CSV file
        df = pd.read_csv(self.raw_csv_path)

        clean_df = df[['raw_input', 'raw_output']].rename(columns={
            'raw_input': 'SMILES',
            'raw_output': 'IUPAC'
        })

        clean_df.dropna(inplace=True)

        # Remove leading and trailing whitespace from 'SMILES' and 'IUPAC'
        clean_df['SMILES'] = clean_df['SMILES'].str.strip()
        clean_df['IUPAC'] = clean_df['IUPAC'].str.strip()

        # Remove rows where 'SMILES' or 'IUPAC' are empty
        clean_df = clean_df[clean_df['SMILES'].str.len() > 0]
        clean_df = clean_df[clean_df['IUPAC'].str.len() > 0]

        # Remove duplicate rows based on 'SMILES' and 'IUPAC'
        clean_df = clean_df.drop_duplicates(subset=['SMILES', 'IUPAC'])

        # Save the cleaned DataFrame to a new CSV file
        clean_df.to_csv(self.csv_path, index=False)

    def load_and_sample_data(self, n=50000):
        """Load CSV and sample n rows."""
        self.df = pd.read_csv(self.csv_path).sample(n=n, random_state=42).reset_index(drop=True)
        print(f"Loaded {len(self.df)} SMILES-IUPAC pairs")

    def build_vocab(self, strings, special_tokens=['<pad>', '<start>', '<end>']):
        """Build a vocabulary from a list of strings."""
        vocab = set()
        for s in strings:
            vocab.update(list(s))
        vocab = sorted(list(vocab))
        vocab = special_tokens + vocab
        idx2char = {i: c for i, c in enumerate(vocab)}
        char2idx = {c: i for i, c in enumerate(vocab)}
        return char2idx, idx2char

    def build_and_save_vocabs(self):
        """Create and save vocabularies for SMILES and IUPAC."""
        self.char2idx_smiles, self.idx2char_smiles = self.build_vocab(self.df['SMILES'])
        self.char2idx_iupac, self.idx2char_iupac = self.build_vocab(self.df['IUPAC'])

        with open(self.smiles_vocab_path, "w") as f:
            json.dump(self.char2idx_smiles, f)
        with open(self.iupac_vocab_path, "w") as f:
            json.dump(self.char2idx_iupac, f)

        print(f"SMILES vocab size: {len(self.char2idx_smiles)}")
        print(f"IUPAC vocab size: {len(self.char2idx_iupac)}")

    def encode_sequence(self, s, vocab, add_bos_eos=False):
        """Encode a string to token indices."""
        tokens = [vocab[c] for c in s if c in vocab]
        if add_bos_eos:
            tokens = [vocab['<start>']] + tokens + [vocab['<end>']]
        return torch.tensor(tokens, dtype=torch.long)

    def encode_all(self):
        """Convert SMILES and IUPAC strings to token tensors."""
        self.smiles_tensors = [self.encode_sequence(s, self.char2idx_smiles) for s in self.df['SMILES']]
        self.iupac_tensors = [self.encode_sequence(s, self.char2idx_iupac, add_bos_eos=True) for s in self.df['IUPAC']]

    def pad_sequences(self):
        """Pad sequences to uniform length for batching."""
        pad_smiles = self.char2idx_smiles['<pad>']
        pad_iupac = self.char2idx_iupac['<pad>']

        self.padded_smiles = pad_sequence(self.smiles_tensors, batch_first=True, padding_value=pad_smiles)
        self.padded_iupac = pad_sequence(self.iupac_tensors, batch_first=True, padding_value=pad_iupac)

        print("Encoded SMILES shape:", self.padded_smiles.shape)
        print("Encoded IUPAC shape:", self.padded_iupac.shape)

    def filter_by_length(self, percentile=95):
        """Filter sequences by length based on a percentile threshold."""
        pad_smiles = self.char2idx_smiles['<pad>']
        pad_iupac = self.char2idx_iupac['<pad>']

        smiles_len = [len(s) for s in self.smiles_tensors]
        iupac_len = [len(i) for i in self.iupac_tensors]

        max_smiles_len = int(np.percentile(smiles_len, percentile))
        max_iupac_len = int(np.percentile(iupac_len, percentile))

        filtered = [(s, i) for s, i in zip(self.smiles_tensors, self.iupac_tensors)
                    if len(s) <= max_smiles_len and len(i) <= max_iupac_len]

        self.padded_smiles = pad_sequence([s for s, _ in filtered], batch_first=True, padding_value=pad_smiles)
        self.padded_iupac = pad_sequence([i for _, i in filtered], batch_first=True, padding_value=pad_iupac)

        print("After filtering:", len(filtered))
        print("New SMILES shape:", self.padded_smiles.shape)
        print("New IUPAC shape:", self.padded_iupac.shape)