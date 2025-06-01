import torch
import os, csv, json, random
import pandas as pd
from tqdm import tqdm

from model.smiles_to_iupac import SMILESToIUPACTransformer
from input_preprocess import SmilesToIUPACPreprocessor
from dataset_preparator import SMILESToIUPACDataModule
from decoder import Decoder
from trainer import Trainer

# -----------------------------
# 1. Configuration Parameters
# -----------------------------
CSV_PATH = 'data/iupac.csv'
VOCAB_SMILES = 'data/smiles_vocab.json'
VOCAB_IUPAC = 'data/iupac_vocab.json'
CHECKPOINT_PATH = 'history/best_model.pth'
LOG_CSV_PATH = 'history/training_log.csv'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
EPOCHS = 50

# Ensure the history directory exists
os.makedirs('history', exist_ok=True)

# -----------------------------
# 2. Preprocess Data
# -----------------------------
preprocessor = SmilesToIUPACPreprocessor(
    csv_path=CSV_PATH,
    vocab_smiles_path=VOCAB_SMILES,
    vocab_iupac_path=VOCAB_IUPAC
)

preprocessor.preprocess_data()
preprocessor.load_and_sample_data(n=50000)
preprocessor.build_and_save_vocabs()
preprocessor.encode_all()
preprocessor.filter_by_length(percentile=95)

# -----------------------------
# 3. Dataset and DataLoader
# -----------------------------
dataModule = SMILESToIUPACDataModule(
    smiles_tensor=preprocessor.padded_smiles,
    iupac_tensor=preprocessor.padded_iupac,
    batch_size=BATCH_SIZE
)
dataModule.setup()
train_loader, val_loader = dataModule.get_loaders()

# -----------------------------
# 4. Initialize Model
# -----------------------------
pad_token_id = preprocessor.char2idx_iupac['<pad>']
start_token_id = preprocessor.char2idx_iupac['<start>']
end_token_id = preprocessor.char2idx_iupac['<end>']

model = SMILESToIUPACTransformer(
    vocab_size_smiles=len(preprocessor.char2idx_smiles),
    vocab_size_iupac=len(preprocessor.char2idx_iupac),
    pad_token_id_smiles=preprocessor.char2idx_smiles['<pad>'],
    pad_token_id_iupac=pad_token_id
).to(DEVICE)

# -----------------------------
# 5. Training
# ----------------------------
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    pad_token_id=pad_token_id,
    lr=1e-4,
    device=DEVICE,
    log_csv_path=LOG_CSV_PATH
)

trainer.train(num_epochs=EPOCHS, checkpoint_path=CHECKPOINT_PATH, early_stopping_patience=10)

# -----------------------------
# 6. Load Best Model
# -----------------------------
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

# -----------------------------
# 7. Decode output
# -----------------------------
decoder = Decoder(
    model=model,
    idx2char=preprocessor.idx2char_iupac,
    start_token_id=start_token_id,
    pad_token_id=pad_token_id,
    end_token_id=end_token_id,
    device=DEVICE
)

df = pd.read_csv(preprocessor.csv_path)
samples = df.sample(15, random_state=42).reset_index(drop=True)

print("Sample predictions:")

inference_csv_path = 'history/inference_results.csv'
with open(inference_csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['SMILES', 'Ground Truth IUPAC', 'Greedy Decoded IUPAC', 'Beam Search Decoded IUPAC'])

    for index, row in tqdm(samples.iterrows(), total=len(samples)):
        smiles = row['SMILES']
        ground_truth_iupac = row['IUPAC']

        # Encode the SMILES string to a tensor
        encoded_input = preprocessor.encode_sequence(smiles, preprocessor.char2idx_smiles)
        input_tensor = torch.tensor(encoded_input, dtype=torch.long).unsqueeze(0).to(DEVICE)

        greedy_output = decoder.greedy_decode(input_tensor)
        beam_output = decoder.beam_search_decode(input_tensor, beam_width=3)

        print("-" * 50)
        print(f"SMILES: {smiles}")
        print(f"Ground Truth IUPAC: {ground_truth_iupac}")
        print(f"Greedy Decoded IUPAC: {greedy_output}")
        print(f"Beam Search Decoded IUPAC: {beam_output}")
        print("-" * 50)

        writer.writerow([smiles, ground_truth_iupac, greedy_output, beam_output])

print(f"See {inference_csv_path} for full results.")

# -----------------------------
# 8. Summary
# -----------------------------
print(f"Training completed. Best model saved at {CHECKPOINT_PATH}")


