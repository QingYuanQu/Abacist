import random
import numpy as np
import torch
import torch.nn as nn

def set_seed(seed=99):
    """设置随机种子以保证实验可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train(model, dataloader, optimizer, device, epochs):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} | Average Loss: {avg_loss:.4f}")

@torch.no_grad()
def generate(model, device, dataset, prompt, max_new):
    model.eval()
    ids = dataset._tokenize(prompt)
    input_ids = torch.tensor([ids], device=device)

    generated = ids[:]
    for _ in range(max_new):
        logits = model(input_ids)
        next_id = logits[0, -1].argmax().item()
        generated.append(next_id)
        input_ids = torch.tensor([generated], device=device)
        char = dataset.itos[next_id]
        yield  char