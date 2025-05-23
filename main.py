import os
from tokenization.character_tokenizer import CharacterTokenizer
from tokenization.dataset_preparator import ImageSMILESDatasetPreparator
from tokenization.image_dataset_builder import ImageToSmilesDatasetBuilder
from model.image_to_smiles import ImageToSmilesModel
from trainer import Trainer

# -----------------------------
# 1. Configuration Parameters
# -----------------------------
MAPPING_JSON_PATH = 'data/image_smiles_mapping.json'
IMAGE_DIR = 'data/molecule_images_handdrawn'
MAX_SEQ_LEN = 100
NUM_SAMPLES = 10000
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 20
CSV_LOG_PATH = 'training_log.csv'
CHECKPOINT_PATH = 'best_model.weights.h5'

# -----------------------------
# 2. Tokenizer Setup
# -----------------------------
print("🔤 Loading tokenizer...")
tokenizer = CharacterTokenizer(mapping_json_path=MAPPING_JSON_PATH)
char_to_idx = tokenizer.char_to_idx
vocab_size = tokenizer.vocab_size

# -----------------------------
# 3. Prepare Paired Data
# -----------------------------
print("🧪 Preparing image-SMILES pairs...")
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
print("🧠 Building model...")
model_builder = ImageToSmilesModel(vocab_size, MAX_SEQ_LEN)
model = model_builder.get_model()
model.summary()

# -----------------------------
# 5. Train Model
# -----------------------------
print("🚀 Starting training...")
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
trainer.train(epochs=EPOCHS, log_csv_path=CSV_LOG_PATH)

print("Training complete. Best model weights saved to:", CHECKPOINT_PATH)
print("Training metrics logged to:", CSV_LOG_PATH)