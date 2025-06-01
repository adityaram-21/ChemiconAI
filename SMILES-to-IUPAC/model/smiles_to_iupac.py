import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
    
class SMILESToIUPACTransformer(nn.Module):
    def __init__(self, vocab_size_smiles, vocab_size_iupac, pad_token_id_smiles, pad_token_id_iupac,
                 d_model=256, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.encoder_embedding = nn.Embedding(vocab_size_smiles, d_model)
        self.decoder_embedding = nn.Embedding(vocab_size_iupac, d_model)

        self.pos_encoder = PositionalEncoding(d_model)
        self.pos_decoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.output_layer = nn.Linear(d_model, vocab_size_iupac)
        
        self.pad_token_id_smiles = pad_token_id_smiles
        self.pad_token_id_iupac = pad_token_id_iupac

    def forward(self, src, tgt):
        src_mask = self.generate_src_mask(src.size(1))
        tgt_mask = self.generate_tgt_mask(tgt.size(1))

        src_emb = self.pos_encoder(self.encoder_embedding(src))
        tgt_emb = self.pos_decoder(self.decoder_embedding(tgt))

        src_emb = src_emb.transpose(0, 1)  # Transformer expects (seq_len, batch, feature)
        tgt_emb = tgt_emb.transpose(0, 1)

        memory = self.encoder(src_emb, src_key_padding_mask = (src == self.pad_token_id_smiles))
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask = (tgt == self.pad_token_id_iupac))

        output = self.output_layer(output)
        return output.transpose(0, 1)
    
    def generate_tgt_mask(self, tgt_len):
        return nn.Transformer.generate_square_subsequent_mask(tgt_len).to(next(self.parameters()).device)
    
    def generate_src_mask(self, src_len):
        return None  # No mask for source in this case, as we use full attention