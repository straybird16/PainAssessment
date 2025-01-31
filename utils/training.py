import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import balanced_accuracy_score

from .models import FocalLoss

def pretrain_transformer(model, train_loader, num_epochs, lr, masking_prob, device, parameters=None):
    """
    Pretrain the transformer using masked sequence modeling.

    Args:
        model (nn.Module): The transformer model.
        train_loader (DataLoader): Data loader for pretraining data.
        num_epochs (int): Number of pretraining epochs.
        lr (float): Learning rate.
        masking_prob (float): Probability of masking each feature.
        device (str): 'cuda' or 'cpu'.

    Returns:
        None
    """
    # Define optimizer and loss function
    parameters = parameters if parameters is not None else model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    criterion = nn.MSELoss()  # Reconstruction loss

    # Move model to device
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for batch_features, _ in train_loader:  # Targets are not used in pretraining
            batch_features = batch_features.to(device)

            # Create a mask for features
            mask = torch.rand(batch_features.shape, device=device) < masking_prob
            masked_features = batch_features.clone()
            masked_features *= mask#masked_features[mask] = 0.0  # Zero out the masked features

            # Forward pass
            reconstructed = model(masked_features)

            # Compute loss only on masked elements
            loss = criterion(reconstructed[mask], batch_features[mask])

            # Backward pass and optimization
            optimizer.zero_grad()
            #scheduler.zero_grad()
            loss.backward()
            scheduler.step(loss)  # Update learning rate based on the loss value
            #optimizer.step()

            train_loss += loss.item()

        print(f"Pretraining Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss / len(train_loader):.4f}")

# Training and evaluation function
def train_transformer(model, train_loader, val_loader, num_epochs, lr, device, parameters=None):
    """
    Train and validate the transformer model.

    Args:
        model (nn.Module): The transformer model.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        num_epochs (int): Number of training epochs.
        lr (float): Learning rate.
        device (str): 'cuda' or 'cpu'.

    Returns:
        None
    """
    # Define optimizer and loss function
    parameters = parameters if parameters is not None else model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    criterion = nn.CrossEntropyLoss(label_smoothing=1e-1)
    #criterion = FocalLoss()
    #criterion = nn.L1Loss()

    # Move model to device
    model.to(device)

    for epoch in range(num_epochs):
        # Training loop
        model.train()
        train_loss = 0
        for batch_features, batch_targets in train_loader:
            batch_features, batch_targets = batch_features.to(device), batch_targets.to(device, dtype=torch.float32)
            
            # Forward pass
            predictions = model(batch_features) # B x L x 1
            # Compute loss
            # use a single segment label (majority voting) instead
            #predictions, batch_targets = predictions.mean(dim=-1), batch_targets.mean(dim=-1) # B x 1
            batch_targets = torch.concatenate((1- batch_targets, batch_targets), -1)
            loss = criterion(predictions, batch_targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            #scheduler.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)    # clip gradients
            scheduler.step(loss)
            #optimizer.step()

            train_loss += loss.item()

        # Validation loop
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        y_pred, y_true = [], []
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features, batch_targets = batch_features.to(device), batch_targets.to(device, dtype=torch.float32)

                # Forward pass
                predictions = model(batch_features) # B x L x 1
                #predictions, batch_targets = predictions.mean(dim=-1), batch_targets.mean(dim=-1) # B x 1
                #predictions = predictions > 0.5 # B x L x 1
                # Compute loss
                batch_targets = torch.concatenate((1- batch_targets, batch_targets), -1)
                val_loss += criterion(predictions, batch_targets).item()

                # Compute accuracy
                #_, predicted = torch.max(predictions, 1)  
                
                #correct += (predictions == batch_targets).sum().item()
                #total += batch_targets.flatten().size(0)
                
                y_pred.append(predictions[:,1].cpu().detach())
                y_true.append(batch_targets[:,1].cpu().detach())
                #predictions, batch_targets = predictions > 0.5, batch_targets > 0.5
        y_pred, y_true = torch.concatenate(y_pred), torch.concatenate(y_true)
        y_pred, y_true = y_pred > 0.5, y_true > 0.5
        average_accuracy = balanced_accuracy_score(y_true.cpu().detach().flatten(), y_pred.cpu().detach().flatten())
        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Train Loss: {train_loss / len(train_loader):.4f}, "
            f"Val Loss: {val_loss / len(val_loader):.4f}, "
            #f"Val Accuracy: {correct / total:.4f}"
            f"Val Accuracy: {average_accuracy :.4f}"
        )
