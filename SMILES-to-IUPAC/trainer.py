import torch
import torch.nn as nn
import csv
import os

class Trainer:
    def __init__(self, model, train_loader, val_loader, pad_token_id, lr=1e-4, device="cuda", log_csv_path="training_log.csv"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.pad_token_id = pad_token_id
        self.device = device
        self.log_csv_path = log_csv_path

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _compute_loss_and_accuracy(self, output, target):
        # Flatten the outputs and targets for loss computation
        loss = self.criterion(output.reshape(-1, output.size(-1)), target.reshape(-1))

        predicted = output.argmax(dim=-1)
        # Calculate accuracy, ignoring padding tokens
        mask = target != self.pad_token_id
        correct = (predicted == target) & mask
        accuracy = correct.sum().item() / mask.sum().item() if mask.sum().item() > 0 else 0

        return loss, accuracy
    
    def train_epoch(self):
        self.model.train()
        total_loss, total_correct, total_tokens = 0, 0, 0

        for batch in self.train_loader:
            src = batch['smiles'].to(self.device)
            target_in = batch['decoder_input'].to(self.device)
            target_out = batch['decoder_target'].to(self.device)

            self.optimizer.zero_grad()
            output = self.model(src, target_in)

            loss, accuracy = self._compute_loss_and_accuracy(output, target_out)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_correct += accuracy * (target_out != self.pad_token_id).sum().item()
            total_tokens += (target_out != self.pad_token_id).sum().item()

        avg_loss = total_loss / len(self.train_loader)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        return avg_loss, accuracy
    
    def validate(self):
        self.model.eval()
        total_loss, total_correct, total_tokens = 0, 0, 0

        with torch.no_grad():
            for batch in self.val_loader:
                src = batch['smiles'].to(self.device)
                target_in = batch['decoder_input'].to(self.device)
                target_out = batch['decoder_target'].to(self.device)

                output = self.model(src, target_in)
                loss, accuracy = self._compute_loss_and_accuracy(output, target_out)

                total_loss += loss.item()
                total_correct += accuracy * (target_out != self.pad_token_id).sum().item()
                total_tokens += (target_out != self.pad_token_id).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        return avg_loss, accuracy
    
    def train(self, num_epochs, checkpoint_path='history/best_model.pth', early_stopping_patience=10):
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if checkpoint_dir and not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        if os.path.exists(self.log_csv_path):
            os.remove(self.log_csv_path)

        with open(self.log_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'])
        
        best_val_accuracy = 0
        best_epoch = 0
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            train_loss, train_accuracy = self.train_epoch()
            val_loss, val_accuracy = self.validate()

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_epoch = epoch
                patience_counter = 0
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"New best model saved at epoch {epoch} with validation accuracy: {val_accuracy:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch} with accuracy {best_val_accuracy:.4f}.")
                    self.model.load_state_dict(torch.load(checkpoint_path))
                    print(f"Restored best model weights from {checkpoint_path}")
                    break

            # Log to CSV
            with open(self.log_csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch, train_loss, val_loss, train_accuracy, val_accuracy])

            print(f"Epoch {epoch:02d}/{num_epochs:02d} - "
                  f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")
    
