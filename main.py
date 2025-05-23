import os
from tokenization.character_tokenizer import CharacterTokenizer
from tokenization.dataset_preparator import ImageSMILESDatasetPreparator
from synthetic_data_generation.dataset_generator import DatasetGenerator
from model.image_to_smiles import ImageToSmilesModel
from trainer import Trainer

# -------------------------------------------------
# 0. Generate Hand-Drawn Images if not already done
# -------------------------------------------------
BASE_DIR = 'data'
VERSION = 'v1'
DATASET_DIR = os.path.join(BASE_DIR, f'handdrawn_{VERSION}')
MAPPING_JSON_PATH = os.path.join(DATASET_DIR, 'metadata.json')
IMAGE_DIR = os.path.join(DATASET_DIR, 'augmented')

if os.path.exists(MAPPING_JSON_PATH) and os.path.exists(IMAGE_DIR) and len(os.listdir(IMAGE_DIR)) > 0:
    print("Hand-drawn images already generated. Skipping generation.")
else:
    print("Generating hand-drawn images...")
    generator = DatasetGenerator(
        smiles_file='GDB13_Subset-ABCDE.smi.gz',
        paper_texture_path='data/paper_texture.png',
        base_dir=BASE_DIR,
        version=VERSION,
        num_samples=100000
    )
    generator.generate()
    print("Hand-drawn images generated and saved to:", IMAGE_DIR)

# -----------------------------
# 1. Configuration Parameters
# -----------------------------
MAX_SEQ_LEN = 100
NUM_SAMPLES = 10000
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 20
CSV_LOG_PATH = 'history/training_log.csv'
CHECKPOINT_PATH = 'history/best_model.weights.h5'
EVALUATION_PATH = 'history/predictions.csv'
MODEL_PATH = 'history/image_to_smiles_model.h5'

# Ensure the history directory exists
os.makedirs('history', exist_ok=True)

# -----------------------------
# 2. Tokenizer Setup
# -----------------------------
print("Loading tokenizer...")
tokenizer = CharacterTokenizer(mapping_json_path=MAPPING_JSON_PATH)
char_to_idx = tokenizer.char_to_idx
vocab_size = tokenizer.vocab_size

# -----------------------------
# 3. Prepare Paired Data
# -----------------------------
print("Preparing image-SMILES pairs...")
preparator = ImageSMILESDatasetPreparator(
    tokenizer=tokenizer,
    input_image_dir=IMAGE_DIR,
    max_len=MAX_SEQ_LEN,
    num_samples=NUM_SAMPLES
)
image_paths, smiles_encoded = preparator.prepare()

# -----------------------------
# 4. Build Model
# -----------------------------
print("Building model...")
model_builder = ImageToSmilesModel(vocab_size, MAX_SEQ_LEN)
model = model_builder.get_model()
model.summary()

# -----------------------------
# 5. Train Model
# -----------------------------
print("Starting training...")
trainer = Trainer(
    model=model,
    image_paths=image_paths,
    tokenized_smiles=smiles_encoded,
    char_to_idx=char_to_idx,
    batch_size=BATCH_SIZE,
    val_split=0.2,
    learning_rate=LEARNING_RATE
)

trainer.setup()
trainer.train(epochs=EPOCHS, log_csv_path=CSV_LOG_PATH, checkpoint_path=CHECKPOINT_PATH)

print("Training complete. Best model weights saved to: ", CHECKPOINT_PATH)
print("Training metrics logged to: ", CSV_LOG_PATH)

#------------------------------
# 6. Evaluate Model
#------------------------------
print("Evaluating model...")
trainer.evaluate(idx_to_char=tokenizer.idx_to_char, num_samples=15, csv_path=EVALUATION_PATH)
print("Evaluation complete. Predictions saved to: ", EVALUATION_PATH)

model.save(MODEL_PATH)
print("Model saved to: ", MODEL_PATH)