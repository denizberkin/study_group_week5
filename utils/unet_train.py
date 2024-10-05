import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, scheduler=None):
    train_losses, val_losses = [], []
    val_precision, val_recall, val_f1 = [], [], []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')

        # Training phase
        model.train()
        running_loss = 0.0
        train_f1 = 0.0
        all_preds_train = []
        all_masks_train = []

        train_bar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}', leave=False)
        
        for batch in train_bar:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()

            loss.backward()
            optimizer.step()

            preds = (torch.sigmoid(outputs) > 0.5).int()
            all_preds_train.append(preds.cpu().numpy().flatten())
            all_masks_train.append(masks.cpu().numpy().flatten())

            batch_preds = np.concatenate(all_preds_train).astype(np.int32)
            batch_masks = np.concatenate(all_masks_train).astype(np.int32)
            train_f1 = f1_score(batch_masks, batch_preds, zero_division=1)

            train_bar.set_postfix({'Loss': running_loss / (train_bar.n + 1), 'F1': train_f1})

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        if scheduler:
            scheduler.step()

        epoch_val_loss, precision, recall, f1 = validate(model, val_loader, criterion, device)
        val_losses.append(epoch_val_loss)
        val_precision.append(precision)
        val_recall.append(recall)
        val_f1.append(f1)

        print(f'Validation: Loss: {epoch_val_loss:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}')

    print('Training complete.')
    return train_losses, val_losses, val_precision, val_recall, val_f1



def validate(model, val_loader, criterion, device):
    """Validation loop to calculate loss, precision, recall, and F1 score."""
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_masks = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating', leave=False):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).int()
            all_preds.append(preds.cpu().numpy().flatten())
            all_masks.append(masks.cpu().numpy().flatten())

    all_preds = np.concatenate(all_preds).astype(np.int32)
    all_masks = np.concatenate(all_masks).astype(np.int32)
    
    precision = precision_score(all_masks, all_preds, zero_division=1)
    recall = recall_score(all_masks, all_preds, zero_division=1)
    f1 = f1_score(all_masks, all_preds, zero_division=1)
    
    epoch_val_loss = val_loss / len(val_loader)

    return epoch_val_loss, precision, recall, f1