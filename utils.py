import torch
from torch.utils.data import DataLoader, TensorDataset
from collate import collate
import gc

def tokens_batch_normalizer(tokens_batch_flat, target_lens) :
    start = 0
    lst = []
    for l in target_lens:
        x = tokens_batch_flat[start: start+l]
        start += l
        lst.append(x)
    return lst
    
def ctc_decode(log_probs, blank_id=0):
    """
    Greedy decode CTC output to token sequence
    
    Args:
        log_probs: (T, batch, vocab_size) - log probabilities
        blank_id: ID of blank token (usually 0)
    
    Returns:
        decoded: List of token sequences for each batch item
    """
    # Get most likely token at each time step
    pred_ids = torch.argmax(log_probs, dim=2)  # (T, batch)
    
    # Decode each batch item
    decoded = []
    for b in range(pred_ids.shape[1]):
        seq = []
        prev = blank_id
        for t in range(pred_ids.shape[0]):
            token = pred_ids[t, b].item()
            # Remove blanks and consecutive duplicates
            if token != blank_id and token != prev:
                seq.append(token)
            prev = token
        decoded.append(seq)
    
    return decoded



from torch.utils.data import DataLoader, TensorDataset
from collate import collate
import gc

class CustomTrainLoader:

    def __init__(self, img_paths, tokens_tensor, chunk_size=10000):

        self.img_paths = img_paths
        self.chunk_size = chunk_size

        self.tokens_all = tokens_tensor
        

    def load_chunk(self, idx):

      
        self.imgs_tensor = torch.load(
            self.img_paths[idx],
            mmap=True,
            weights_only=False
        )
      
        if idx == 4 :
            start = idx  * self.chunk_size
            self.tokens = self.tokens_all[start:]
        else:
            start = idx  * self.chunk_size
            end = (idx + 1) * self.chunk_size
            self.tokens = self.tokens_all[start:end]

        assert len(self.imgs_tensor) == len(self.tokens)

        
        self.dataset = TensorDataset(self.imgs_tensor, self.tokens)

        self.loader = DataLoader(
            self.dataset,
            batch_size=128,
            shuffle=True,
            collate_fn=collate
        )
        return self.loader


    def release(self):

        del self.loader
        del self.imgs_tensor
        del self.tokens
        del self.dataset

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
