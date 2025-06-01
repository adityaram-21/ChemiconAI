import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class SMILESToIUPACDataset(Dataset):
    def __init__(self, smiles_tensor, iupac_tensor):
        self.smiles = smiles_tensor
        self.iupac = iupac_tensor

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        # Decoder input: exclude the last token
        decoder_input = self.iupac[idx][:-1]

        # Decoder target: exclude the first token
        decoder_target = self.iupac[idx][1:]
        return {
            'smiles': self.smiles[idx],
            'decoder_input': decoder_input,
            'decoder_target': decoder_target
        }
    
class SMILESToIUPACDataModule:
    def __init__(self, smiles_tensor, iupac_tensor, batch_size=32, val_split=0.2, seed=42):
        self.smiles_tensor = smiles_tensor
        self.iupac_tensor = iupac_tensor
        self.batch_size = batch_size
        self.val_split = val_split
        self.seed = seed

        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None

    def setup(self):
        """Prepare the dataset by splitting into train and validation sets."""
        train_smiles, val_smiles, train_iupac, val_iupac = train_test_split(
            self.smiles_tensor, self.iupac_tensor, test_size=self.val_split, random_state=self.seed
        )

        self.train_dataset = SMILESToIUPACDataset(train_smiles, train_iupac)
        self.val_dataset = SMILESToIUPACDataset(val_smiles, val_iupac)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def get_loaders(self):
        """Return the train and validation data loaders."""
        if self.train_loader is None or self.val_loader is None:
            raise RuntimeError("DataModule not set up. Call setup() first.")
        return self.train_loader, self.val_loader