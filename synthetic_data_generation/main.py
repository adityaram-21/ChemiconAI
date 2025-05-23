from dataset_generator import HandDrawnDatasetGenerator

generator = HandDrawnDatasetGenerator(
    smiles_file='GDB13_Subset-ABCDE.smi.gz',
    paper_texture_path='data/paper_texture.png',
    base_dir='data',
    version='v1',
    num_samples=100000
)
generator.generate()