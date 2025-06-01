import os, cv2, random, json
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image, ImageFilter, ImageOps
from collections import Counter

class HandDrawnDatasetGenerator:
    def __init__(self,
                 smiles_file = '../data/smiles/GDB13_Subset-ABCDE.smi.gz',
                 paper_texture_path = '../data/texture/paper_texture.png',
                 base_dir = 'dataset_v1',
                 num_samples = 100000,
                 top_N = 14000000,
                 version = 'v1',
                 seed = 42):
        
        self.smiles_file = smiles_file
        self.paper_texture_path = paper_texture_path
        self.num_samples = num_samples
        self.top_N = top_N
        self.version = version
        self.seed = seed

        # Directories
        self.base_dir = os.path.join(base_dir, f'handdrawn_{version}')
        self.image_dir = os.path.join(self.base_dir, 'images')
        self.augmented_dir = os.path.join(self.base_dir, 'augmented')
        self.log_path = os.path.join(self.base_dir, 'metadata.json')

        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.augmented_dir, exist_ok=True)

        random.seed(seed)
        np.random.seed(seed)

        self.metadata = []
    
    def load_smiles(self):
        """
        Load samples from SMILES strings.
        
        Returns:
        DataFrame: A DataFrame containing the loaded samples.
        """
        # Create a DataFrame from the SMILES strings
        smiles_data = pd.read_csv(self.smiles_file, names=['SMILES'], compression='gzip', nrows=self.top_N)
        # Generate a random number of samples
        self.sampled_smiles = smiles_data.sample(n=self.num_samples, random_state=self.seed).reset_index(drop=True) 
        return self.sampled_smiles
    
    def draw_molecules(self):
        """
        Draw the molecules from the sampled SMILES strings and save them as images.
        """

        for idx, row in self.sampled_smiles.iterrows():
            mol = Chem.MolFromSmiles(row['SMILES'])
            if mol is not None:
                img = Draw.MolToImage(mol, size=(256, 256))  # You can adjust size here
                img_path = os.path.join(self.image_dir, f'mol_{idx}.png')
                img.save(img_path)
                self.metadata.append({
                    'id': idx,
                    'SMILES': row['SMILES'],
                    'image_path': img_path,
                    'augmented': False
                })
            else:
                print(f"Invalid SMILES: {row['SMILES']}")

            # Limit to 100,000 images
            if idx >= self.num_samples - 1:
                break
    
    def simulate_hand_drawn(self, blend_alpha = 0.05):
        """
        Simulate hand-drawn effect on the images.
        
        Parameters:
        blend_alpha (float): Alpha value for blending the paper texture (default is 0.1).
        """
        for entry in self.metadata:
            input_path = entry['image_path']
            img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Apply simple sketching
            blurred = cv2.GaussianBlur(img, (5, 5), 1)
            _, sketch = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)

            # Morphological noise removal
            kernel = np.ones((2, 2), np.uint8)
            opened = cv2.morphologyEx(sketch, cv2.MORPH_OPEN, kernel)

            # Mild distortion
            rows, cols = opened.shape
            dx = np.random.randint(-1, 1, (rows, cols)).astype(np.float32)
            dy = np.random.randint(-1, 1, (rows, cols)).astype(np.float32)
            map_x, map_y = np.meshgrid(np.arange(cols), np.arange(rows))
            map_x = (map_x + dx).astype(np.float32)
            map_y = (map_y + dy).astype(np.float32)
            warped = cv2.remap(opened, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderValue=255)

            # Convert to PIL and simulate sketch
            pil_img = Image.fromarray(255 - warped)
            pil_img = pil_img.rotate(random.uniform(-3, 3), expand=True, fillcolor=255)

            # Apply paper texture if provided
            if self.paper_texture_path:
                texture = Image.open(self.paper_texture_path).convert("L").resize(pil_img.size)
                pil_img = Image.blend(pil_img, texture, alpha=blend_alpha)

            pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.6))
            pil_img = ImageOps.autocontrast(pil_img, cutoff=2)

            aug_path = os.path.join(self.augmented_dir, os.path.basename(input_path).replace('.png', '_handdrawn.png'))
            pil_img.save(aug_path)

            # Update metadata
            entry['augmented'] = True
            entry['augmented_image'] = aug_path

    def save_metadata(self):
        with open(self.log_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def generate(self):
        """
        Main method to generate the dataset.
        """
        # Load SMILES
        self.load_smiles()
        # Draw molecules
        self.draw_molecules()
        # Simulate hand-drawn effect
        self.simulate_hand_drawn()
        # Save metadata
        self.save_metadata()

        print(f"Dataset generated at {self.base_dir} with {self.num_samples} samples.")