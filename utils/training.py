
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, matthews_corrcoef, roc_auc_score

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
            #scheduler.step(loss)  # Update learning rate based on the loss value
            optimizer.step()

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
    #optimizer = torch.optim.SGD(parameters, lr=lr, nesterov=True, momentum=0.9)
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0e-1)
    #criterion = FocalLoss()
    #criterion = nn.L1Loss()

    # Move model to device
    model.to(device)

    for epoch in range(num_epochs):
        # Training loop
        model.train()
        train_loss = 0
        y_pred, y_true = [], []
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
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)    # clip gradients
            #scheduler.step(loss)
            optimizer.step()
            train_loss += loss.item()
            predictions, batch_targets = predictions[:,1], batch_targets[:,1]
            y_pred.append(predictions.cpu().detach())
            y_true.append(batch_targets.cpu().detach())
        y_pred, y_true = torch.concatenate(y_pred).cpu().detach().flatten(), torch.concatenate(y_true).cpu().detach().flatten()
        train_auc = roc_auc_score(y_true > 0.5, y_pred)
        y_pred, y_true = y_pred > 0.5, y_true > 0.5
        train_average_accuracy, train_mcc, train_accuracy = balanced_accuracy_score(y_true, y_pred), matthews_corrcoef(y_true, y_pred), accuracy_score(y_true, y_pred)
        train_f1_pos, train_f1_neg = f1_score(y_true, y_pred), f1_score(~y_true, ~y_pred)
        train_f1_m = (train_f1_pos + train_f1_neg) / 2
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
                predictions, batch_targets = predictions[:,1], batch_targets[:,1]
                y_pred.append(predictions.cpu().detach())
                y_true.append(batch_targets.cpu().detach())
                #predictions, batch_targets = predictions > 0.5, batch_targets > 0.5
        y_pred, y_true = torch.concatenate(y_pred).cpu().detach().flatten(), torch.concatenate(y_true).cpu().detach().flatten()
        val_auc = roc_auc_score(y_true > 0.5, y_pred)
        y_pred, y_true = y_pred > 0.5, y_true > 0.5
        val_average_accuracy, val_mcc, val_accuracy = balanced_accuracy_score(y_true, y_pred), matthews_corrcoef(y_true, y_pred), accuracy_score(y_true, y_pred)
        val_f1_pos, val_f1_neg = f1_score(y_true, y_pred), f1_score(~y_true, ~y_pred)
        val_f1_m = (val_f1_pos + val_f1_neg) / 2

        #train_average_accuracy = _get_balanced_accuracy(model, train_loader, dev)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Train Loss: {train_loss / len(train_loader):.4f}, "
            f"Val Loss: {val_loss / len(val_loader):.4f}, "
            #f"Val Accuracy: {correct / total:.4f}"
            f"Train accuracy: {train_accuracy :.4f},"
            f"Train auc: {train_auc :.4f}, "
            f"Train mcc: {train_mcc :.4f}, \n\t"
            f"Train F1_m: {train_f1_m :.4f}, "
            f"Train average accuracy: {train_average_accuracy :.4f}, \n\t"
            f"Val accuracy: {val_accuracy :.4f},"
            f"Val auc: {val_auc :.4f}, "
            f"Val mcc: {val_mcc :.4f}, \n\t"
            f"Val F1_m: {val_f1_m :.4f}, "
            f"Val average accuracy: {val_average_accuracy :.4f}, \n\t"
        )


def train_multi_task_model(
    model, train_dataloaders, val_dataloaders, task_list, num_epochs, lr_main:float|dict=1e-4, lr_head:float|dict=1e-3, device="cuda"
):
    """
    Train a multi-task transformer model with both training and validation phases.

    Args:
        model (nn.Module): Multi-task transformer model with `set_task()`.
        train_dataloaders (dict): Dataloaders for training tasks.
        val_dataloaders (dict): Dataloaders for validation tasks.
        task_list (list): List of tasks to train in each epoch.
        num_epochs (int): Number of training epochs.
        lr_main (float): Small learning rate for backbone encoder.
        lr_head (float): Learning rate for task-specific heads.
        device (str): Device to train on ("cuda" or "cpu").

    Returns:
        pd.DataFrame: Training and validation losses and behavior detection metrics for each epoch.
    """
    valid_tasks =  [
            "predict_joint", "reconstruct_kinematics", "predict_pain",
            "predict_behavior", "predict_activity"
        ]
    model.to(device)
    criterion_mse = nn.MSELoss()
    criterion_ce = nn.CrossEntropyLoss()
    #criterion_ce = FocalLoss()

    # Optimizers: Small LR for encoder, normal LR for task-specific heads
    lr_main = lr_main if type(lr_main) is dict else {task:lr_main for task in valid_tasks}
    lr_head = lr_head if type(lr_head) is dict else {task:lr_head for task in valid_tasks}
    optimizer_main = {
        task: optim.Adam(model.backbone.parameters(), lr=lr_main[task]) for task in valid_tasks
    }
    optimizer_heads = {
        task: optim.Adam(model.get_task_modules(task).parameters(), lr=lr_head[task]) for task in valid_tasks
    }

    # DataFrame to store losses and behavior detection metrics
    history = pd.DataFrame(columns=[
        "epoch", "train_joint_loss", "val_joint_loss", "train_reconstruction_loss", "val_reconstruction_loss",
        "train_pain_loss", "val_pain_loss", "train_behavior_loss", "val_behavior_loss",
        "train_activity_loss", "val_activity_loss", "train_behavior_accuracy", "val_behavior_accuracy",
        "train_behavior_f1_positive", "val_behavior_f1_positive", "train_behavior_f1_mean", "val_behavior_f1_mean",
        "train_behavior_mcc", "val_behavior_mcc", "train_behavior_auc", "val_behavior_auc"
    ])

    def run_epoch(dataloaders, train=True):
        """Runs an epoch for either training or validation phase."""
        model.train() if train else model.eval()
        phase = "train" if train else "val"
        task_losses = {task: 0.0 for task in task_list}
        all_preds, all_labels = [], []

        for task in task_list:
            if task not in dataloaders:
                continue

            dataloader = dataloaders[task]
            model.set_task(task)

            for batch in dataloader:
                batch = [b.to(device) for b in batch]
                kinematics = batch[0]

                optimizer_heads[task].zero_grad()
                optimizer_main[task].zero_grad()

                with torch.set_grad_enabled(train):  # Disable gradient for validation
                    outputs = model(kinematics)

                    if task == "predict_joint":
                        loss = criterion_mse(outputs, batch[1])

                    elif task == "reconstruct_kinematics":
                        loss = criterion_mse(outputs, batch[1])

                    elif task == "predict_pain":
                        loss = criterion_ce(outputs, batch[1])
                    elif task == "predict_behavior":
                        behavior_targets = batch[1]
                        behavior_targets = torch.concatenate((1- behavior_targets, behavior_targets), -1)
                        #print(outputs.shape, behavior_targets.shape)
                        loss = criterion_ce(outputs, behavior_targets)
                        # Collect predictions & labels for behavior metrics
                        preds = torch.argmax(outputs, dim=-1).view(-1).cpu().numpy()
                        labels = torch.argmax(behavior_targets, dim=-1).view(-1).cpu().numpy()
                        #labels = behavior_targets.view(-1).cpu().numpy()
                        all_preds.extend(preds)
                        all_labels.extend(labels)

                    elif task == "predict_activity":
                        loss = criterion_ce(outputs, batch[1])

                    # Backprop only during training
                    if train:
                        loss.backward()
                        optimizer_heads[task].step()
                        optimizer_main[task].step()

                    task_losses[task] += loss.item()/len(dataloader)

        # Compute behavior detection metrics
        behavior_metrics = {"accuracy": None, "f1_positive": None, "f1_mean": None, "mcc": None, "auc": None}
        if "predict_behavior" in task_list:
            behavior_metrics["accuracy"] = accuracy_score(all_labels, all_preds)
            behavior_metrics["balanced_accuracy"] = balanced_accuracy_score(all_labels, all_preds)
            behavior_metrics["f1_positive"] = f1_score(all_labels, all_preds, pos_label=1)
            behavior_metrics["f1_mean"] = f1_score(all_labels, all_preds, average="macro")
            behavior_metrics["mcc"] = matthews_corrcoef(all_labels, all_preds)
            behavior_metrics["auc"] = roc_auc_score(all_labels, all_preds, multi_class="ovr")

        return task_losses, behavior_metrics

    for epoch in range(num_epochs):
        train_losses, train_behavior_metrics = run_epoch(train_dataloaders, train=True)
        val_losses, val_behavior_metrics = run_epoch(val_dataloaders, train=False)

        # Create a dictionary for logging this epoch
        epoch_data = {
            "epoch": epoch + 1,
            "train_joint_loss": train_losses.get("predict_joint"),
            "val_joint_loss": val_losses.get("predict_joint"),
            "train_reconstruction_loss": train_losses.get("reconstruct_kinematics"),
            "val_reconstruction_loss": val_losses.get("reconstruct_kinematics"),
            "train_pain_loss": train_losses.get("predict_pain"),
            "val_pain_loss": val_losses.get("predict_pain"),
            "train_behavior_loss": train_losses.get("predict_behavior"),
            "val_behavior_loss": val_losses.get("predict_behavior"),
            "train_activity_loss": train_losses.get("predict_activity"),
            "val_activity_loss": val_losses.get("predict_activity"),
            "train_behavior_accuracy": train_behavior_metrics["accuracy"],
            "val_behavior_accuracy": val_behavior_metrics["accuracy"],
            "train_behavior_balanced_accuracy": train_behavior_metrics["balanced_accuracy"],
            "val_behavior_balanced_accuracy": val_behavior_metrics["balanced_accuracy"],
            "train_behavior_f1_positive": train_behavior_metrics["f1_positive"],
            "val_behavior_f1_positive": val_behavior_metrics["f1_positive"],
            "train_behavior_f1_mean": train_behavior_metrics["f1_mean"],
            "val_behavior_f1_mean": val_behavior_metrics["f1_mean"],
            "train_behavior_mcc": train_behavior_metrics["mcc"],
            "val_behavior_mcc": val_behavior_metrics["mcc"],
            "train_behavior_auc": train_behavior_metrics["auc"],
            "val_behavior_auc": val_behavior_metrics["auc"],
        }

        # Append new row using pd.concat()
        history = pd.concat([history, pd.DataFrame([epoch_data])], ignore_index=True)

        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs}: "
                f"Train Loss - Joint: {_format_loss(train_losses.get('predict_joint', 'N/A'))}, "
                f"Reconstruction: {_format_loss(train_losses.get('reconstruct_kinematics', 'N/A'))}, "
                f"Pain: {_format_loss(train_losses.get('predict_pain', 'N/A'))}, "
                f"Behavior: {_format_loss(train_losses.get('predict_behavior', 'N/A'))}, "
                f"Behavior balanced accuracy: {_format_loss(train_behavior_metrics.get('balanced_accuracy', 'N/A'))}, "
                f"Activity: {_format_loss(train_losses.get('predict_activity', 'N/A'))} || "
                f"Val Loss - Joint: {_format_loss(val_losses.get('predict_joint', 'N/A'))}, "
                f"Reconstruction: {_format_loss(val_losses.get('reconstruct_kinematics', 'N/A'))}, "
                f"Pain: {_format_loss(val_losses.get('predict_pain', 'N/A'))}, "
                f"Behavior: {_format_loss(val_losses.get('predict_behavior', 'N/A'))}, "
                f"Behavior balanced accuracy: {_format_loss(val_behavior_metrics.get('balanced_accuracy', 'N/A'))}, "
                f"Activity: {_format_loss(val_losses.get('predict_activity', 'N/A'))}")

    return history

def _format_loss(value):
    """Formats loss values to 4 decimal places, or returns 'N/A' if not available."""
    return f"{value:.4f}" if isinstance(value, (int, float)) else "N/A"