import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import csv

class Trainer:
    def __init__(self, model, image_paths, tokenized_smiles, char_to_idx,
                 batch_size=32, val_split=0.2, learning_rate=1e-4):
        
        self.model = model
        self.image_paths = image_paths
        self.tokenized_smiles = tokenized_smiles
        self.char_to_idx = char_to_idx
        self.batch_size = batch_size
        self.val_split = val_split
        self.learning_rate = learning_rate

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, reduction='none'
        )

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.PAD_TOKEN_IDX = self.char_to_idx['<pad>']
        self.train_dataset = None
        self.val_dataset = None

    def _masked_loss(self, y_true, y_pred):
        loss_ = self.loss_object(y_true, y_pred)
        mask = tf.cast(tf.not_equal(y_true, self.PAD_TOKEN_IDX), dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    def _process_image(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, (256, 256))
        return img

    def _prepare_tf_dataset(self, img_paths, smiles):
        def map_fn(image_path, smile_seq):
            img = self._process_image(image_path)
            decoder_input = smile_seq[:-1]
            decoder_target = smile_seq[1:]
            return (img, decoder_input), decoder_target

        dataset = tf.data.Dataset.from_tensor_slices((img_paths, smiles))
        dataset = dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(1000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    def setup(self):
        # Split into train and validation sets
        train_img_paths, val_img_paths, train_smiles, val_smiles = train_test_split(
            self.image_paths, self.tokenized_smiles, test_size=self.val_split, random_state=42)

        self.train_dataset = self._prepare_tf_dataset(train_img_paths, train_smiles)
        self.val_dataset = self._prepare_tf_dataset(val_img_paths, val_smiles)

        # Compile the model with masked loss
        self.model.compile(optimizer=self.optimizer, loss=self._masked_loss, metrics=['accuracy'])

        print("Model compiled and datasets prepared.")

    def train(self, epochs=20, log_csv_path='training_log.csv', checkpoint_path='best_model.weights.h5'):
        if self.train_dataset is None or self.val_dataset is None:
            raise ValueError("Call `.setup()` before training.")
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        checkpoint = ModelCheckpoint(filepath = checkpoint_path, save_best_only=True, monitor='val_loss', mode='min')

        history = self.model.fit(self.train_dataset, validation_data=self.val_dataset, epochs=epochs, callbacks=[early_stop, checkpoint])

        # Save training history to CSV
        keys = history.history.keys()
        with open(log_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(keys)
            writer.writerows(zip(*[history.history[k] for k in keys]))

        print(f"Training metrics saved to '{log_csv_path}'.")
        return history
    
    def evaluate(self, idx_to_char, num_samples=15, csv_path='predictions.csv'):
        if self.val_dataset is None:
            raise ValueError("Validation dataset not prepared. Call `.setup()` first.")

        for (img_batch, decoder_input), target in self.val_dataset.take(1):
            predictions = self.model.predict([img_batch, decoder_input])
            predicted_ids = tf.argmax(predictions, axis=-1).numpy()

            for i in range(min(num_samples, len(predicted_ids))):
                pred_tokens = [idx_to_char[idx] for idx in predicted_ids[i] if idx_to_char[idx] != '<pad>']
                target_tokens = [idx_to_char[idx] for idx in target[i].numpy() if idx_to_char[idx] != '<pad>']

                print(f"\nSample {i + 1}")
                print("Predicted SMILES:", ''.join(pred_tokens))
                print("Target SMILES   :", ''.join(target_tokens))

                #copy to csv
                with open(csv_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([i + 1, ''.join(pred_tokens), ''.join(target_tokens)])