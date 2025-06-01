import torch
import heapq
import torch.nn.functional as F

class Decoder:
    def __init__(self, model, idx2char, start_token_id, pad_token_id, end_token_id, device='cuda'):
        self.model = model.to(device)
        self.idx2char = idx2char
        self.start_token_id = start_token_id
        self.pad_token_id = pad_token_id
        self.end_token_id = end_token_id
        self.device = device

    def decode(self, encoded_sequence):
        """Decode a sequence of token indices to a string."""
        tokens = []
        for idx in encoded_sequence:
            idx = idx.item()
            token = self.idx2char.get(str(idx), self.idx2char.get(idx, None))
            if token is None:
                continue
            if token == '<end>':
                break
            if token not in ['<pad>', '<start>']:
                tokens.append(token)
        return ''.join(tokens)
    
    def greedy_decode(self, input_tensor, max_length=512):
        """Greedily decode a sequence."""
        self.model.eval()
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device).unsqueeze(0) if input_tensor.dim() == 1 else input_tensor.to(self.device)
            target_tensor = torch.tensor([[self.start_token_id]], dtype=torch.long).to(self.device)

            for _ in range(max_length):
                output = self.model(input_tensor, target_tensor)
                next_token = output.argmax(dim=-1)[:, -1]
                target_tensor = torch.cat([target_tensor, next_token], dim=1)
                if next_token.item() == self.end_token_id:
                    break
            return self.decode(target_tensor.squeeze(0))
        
    def beam_search_decode(self, input_tensor, beam_width=3, max_length=512):
        """Beam search decoding."""
        self.model.eval()
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device).unsqueeze(0) if input_tensor.dim() == 1 else input_tensor.to(self.device)
            sequences = [(0.0, [self.start_token_id])]  # (score, sequence)
            for _ in range(max_length):
                all_candidates = []
                for score, seq in sequences:
                    target_tensor = torch.tensor([seq], dtype=torch.long).to(self.device)
                    output = self.model(input_tensor, target_tensor)
                    probs = F.log_softmax(output[:, -1, :], dim=-1)
                    top_probs, top_indices = torch.topk(probs, beam_width)

                    for i in range(beam_width):
                        next_seq = seq + [top_indices[0, i].item()]
                        next_score = score + top_probs[0, i].item()
                        all_candidates.append((next_score, next_seq))

                # Select the best sequences
                sequences = heapq.nlargest(beam_width, all_candidates, key=lambda x: x[0])

                # Early stopping if all sequences end with <end>
                if all(seq[-1] == self.end_token_id for _, seq in sequences):
                    break
            
            return self.decode(torch.tensor(sequences[0][1], dtype=torch.long).to(self.device))