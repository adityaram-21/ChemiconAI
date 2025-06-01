from character_tokenizer import CharacterTokenizer
import os

class ImageSMILESDatasetPreparator:
    def __init__(self, tokenizer: CharacterTokenizer, input_image_dir, max_len=100, num_samples=10000):
        self.tokenizer = tokenizer
        self.input_image_dir = input_image_dir
        self.max_len = max_len
        self.num_samples = num_samples

        self.image_paths = []
        self.smiles_encoded = []

    def prepare(self):
        for idx, smile in enumerate(self.tokenizer.smiles_list[:self.num_samples]):
            img_path = os.path.join(self.input_image_dir, f"mol_{idx}_handdrawn.png")
            if os.path.exists(img_path):
                self.image_paths.append(img_path)
                self.smiles_encoded.append(self.tokenizer.encode(smile, max_len=self.max_len))

        print(f"Prepared {len(self.image_paths)} image-text pairs.")
        print("Example Image Path:", self.image_paths[0])
        print("Original SMILES:", self.tokenizer.smiles_list[0])
        print("Tokenized SMILES:", self.smiles_encoded[0])
        print("Decoded:", self.tokenizer.decode(self.smiles_encoded[0]))

        return self.image_paths, self.smiles_encoded