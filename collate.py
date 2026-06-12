import torch

def collate(batch):
    img_batch = []
    tokens_batch = []
    tokens_lengths = []


    for img, token in batch:
        img_batch.append(img)
        depadded_token = torch.tensor([t for t in token if t]) #selecting the nonzero values, depadding
        tokens_batch.append(depadded_token)
        tokens_lengths.append(len(depadded_token))

    img_batch = torch.stack(img_batch)
    ctc_tokens = torch.cat(tokens_batch)
    target_length = torch.tensor(tokens_lengths)


    return img_batch, ctc_tokens, target_length
    

    

