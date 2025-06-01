import tensorflow as tf

class ImageToSmilesDatasetBuilder:
    def __init__(self, image_paths, tokenized_smiles, 
                 img_size=(256, 256), max_seq_len=100, batch_size=32):
        self.image_paths = image_paths
        self.tokenized_smiles = tokenized_smiles
        self.img_size = img_size
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

    def _process_image(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img, channels=1)  # grayscale
        img = tf.image.convert_image_dtype(img, tf.float32)  # scale to 0–1
        img = tf.image.resize(img, self.img_size)
        return img

    def _load_data(self, image_path, smile_seq):
        img = self._process_image(image_path)

        # Split sequence: decoder_input = seq[:-1], decoder_target = seq[1:]
        decoder_input = smile_seq[:-1]
        decoder_target = smile_seq[1:]

        return (img, decoder_input), decoder_target

    def build_dataset(self, shuffle_buffer=1000):
        # Convert lists to tensors
        image_paths_tensor = tf.constant(self.image_paths)
        smiles_tensor = tf.constant(self.tokenized_smiles)

        # Build pipeline
        dataset = tf.data.Dataset.from_tensor_slices((image_paths_tensor, smiles_tensor))
        dataset = dataset.map(self._load_data, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(shuffle_buffer)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset