import numpy as np
import pandas as pd
import pickle

import torch
from torchvision import models
import torch.optim as optim
import torch.nn as nn

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

device = "cpu"

#Instantiate pretrained resnet18 model with 5 classes
def create_resnet18(num_classes=5):

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    model.fc = nn.Linear(model.fc.in_features, num_classes) #replace fc by 5 heads
    
    return model

#Load existing saved moel
def load_model(path):
    
    model = create_resnet18(5) #replace fc by 5 heads
    
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    
    return model

#Freeze all layers except fc head (final fully connected classifier)
def freeze_backbone(model):
    
    for name, param in model.named_parameters():
        
        param.requires_grad = name.startswith("fc") #only train fc head...nothing else

#unfreeze all parameters
def unfreeze_all(model):
    
    for param in model.parameters():
        param.requires_grad = True #train params

#AdamW optimizer to train params
def set_optimizer(model, lr):
    params = (param for param in model.parameters() if param.requires_grad)
    return optim.AdamW(params, lr=lr)

#Run 1 train epoch
#Return avg_loss and accuracy
def train_one_epoch(model, optimizer, train_loader, criterion):

    #Train 1 epoch
    model.train()

    #Keep track of correct predictions
    total_correct, total = 0, 0
    total_loss = 0.0

    for images, labels in train_loader:
        
        images, labels = images, labels

        #Reset gradient
        optimizer.zero_grad()

        #raw model weights
        #forward pass + compute loss
        logits = model(images)
        loss = criterion(logits, labels)

        #backprop
        loss.backward()

        #update weights
        optimizer.step()

        #Predict
        preds = logits.argmax(dim=1)

        #Evaluate predictions
        
        total_correct += (preds == labels).sum().item()

        #batch loss
        total_loss += (loss.item() * len(labels))

        #total number of samples
        total += len(labels)

        avg_loss = total_loss / total
        avg_acc = total_correct / total

    #Avg loss, Accuracy
    return avg_loss, avg_acc 

#Same thing as train epoch but for validation
#Return loss, accuracy, AND PREDICTIONS

def val_one_epoch(model, val_loader, criterion):

    #Turn off dropout
    model.eval()

    #correct preds, total samples seen
    total_correct, total = 0, 0
    total_loss = 0.0

    #accumulators for predictions (batch wise)
    preds_all, labels_all = [], []

    #No backprop + quicker inference
    with torch.no_grad():
        
        for images, labels in val_loader:
            images, labels = images, labels

            #Raw model weights
            #Forward pass
            logits = model(images)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1) #convert logits to class

            total_correct += (preds == labels).sum().item()
            total_loss += loss.item() * len(labels)
            total += len(labels)

            preds_all.append(preds.cpu())
            labels_all.append(labels.cpu())

            avg_val_loss = total_loss / total
            val_acc =total_correct / total
            

    return (avg_val_loss, val_acc, torch.cat(preds_all), torch.cat(labels_all) )


def train_model(seed, save_name, train_loader, val_loader, criterion):
    """
    Two-phase training:
    
        Phase 1: Convolutional Backbone frozen: train only the final classification layer.
        Doesn't destroy  pretrained weights
        
        Phase 2: unfreeze the entire network and fine-tune.
        Smaller learning rate + early stopping
        
    """
    
    #set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    #metrics
    eval_epoch = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
    "stop_epoch": None
}


    #Replace final classifier head by 5 classes
    model = create_resnet18(5)

    print(f"\nTRAINING RESNET18 with SEED {seed}")

    #PHASE 1: Freeze everything except fc head
    #High learning rate compared to phase 2 since only last layer is trained
    freeze_backbone(model)
    optimizer = set_optimizer(model, lr=1e-3)

    best_acc = 0.0
    final_preds = None
    final_labels = None

    #5 epochs only
    for epoch in range(5):
        print(f"\nPhase 1 — Epoch {epoch+1}/5")

        #Train epoch
        train_loss, train_acc = train_one_epoch(model, optimizer, train_loader, criterion)

        #Validation epoch
        val_loss, val_acc, all_preds, all_labels = val_one_epoch(model, val_loader, criterion)

        eval_epoch["train_loss"].append(train_loss)
        eval_epoch["train_acc"].append(train_acc)
        eval_epoch["val_loss"].append(val_loss)
        eval_epoch["val_acc"].append(val_acc)

        print(f"Train Acc: {train_acc:.3f} | Train Loss: {train_loss:.3f}")
        print(f"Val   Acc: {val_acc:.3f} | Val   Loss: {val_loss:.3f}")

        #Save best moel yet as checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_name)

    #Phase 2: Unfreeze the whole network + FineTune
    #All layers learn
    #optimizer with smaller learning rate
    
    unfreeze_all(model)
    optimizer = set_optimizer(model, lr=1e-4)

    #stop early if no improvement for 5 epochs
    tolerance, curr_tolerance = 5, 0
    
    for epoch in range(25):
        
        print(f"\nPhase 2 — Epoch {epoch+1}/25")

        #Train epoch
        train_loss, train_acc = train_one_epoch(model, optimizer, train_loader, criterion)
        
        #Validation epoch
        val_loss, val_acc, final_preds, final_labels = val_one_epoch(model, val_loader, criterion)

        eval_epoch["train_loss"].append(train_loss)
        eval_epoch["train_acc"].append(train_acc)
        eval_epoch["val_loss"].append(val_loss)
        eval_epoch["val_acc"].append(val_acc)

        print(f"Train Acc: {train_acc:.3f} | Train Loss: {train_loss:.3f}")
        print(f"Val Acc: {val_acc:.3f} | Val Loss: {val_loss:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_name)
            curr_tolerance = 0
            
        else:
            curr_tolerance += 1
            if curr_tolerance >= tolerance:
                print("Stopped Early.")
                eval_epoch["stop_epoch"] = (epoch + 1)  #1-based indexing
                break
    
    #End of for epochs loop

    #If no early stopping, set to last epoch
    if eval_epoch["stop_epoch"] is None:
        eval_epoch["stop_epoch"] = len(eval_epoch["train_loss"])
        
    print(f"Best accuracy for seed {seed}: {best_acc:.3f}")
    print_final_report(final_preds.cpu(), final_labels.cpu())
    
    return eval_epoch, final_preds, final_labels

#Loads checkpoints compatible with old or new resnet formats -> we use a mixture of models for this competition
#Reusable function
def load_resnet18_checkpoint(path, num_classes=5):
    """
    Universal loader for ResNet-18 checkpoints.

    Deals with:
        old-style checkpoints with 'model.' prefix on all keys
        checkpoints from create_resnet18
        Any future variant using torch ResNet18 key structure
    """
    #normal definition
    model = create_resnet18(num_classes)

    #Load raw state dict
    state = torch.load(path, map_location=device)

    #Create a new dict with sanitized key names
    new_state = {}
    for k, v in state.items():
        
        #If checkpoint was saved using ResNet18Classifier:
        #keys are like "model.conv1.weight"
        
        if k.startswith("model."):
            
            k = k[len("model."):]  #remove "model."
            
        new_state[k] = v

    #Load modified keys
    model.load_state_dict(new_state, strict=True)
    model.eval()
    
    return model

#models: list of torch models
def final_pred_ensemble(models, test_loader, device="cpu"):
    """
    run ensemble inference over a test dataloader
    each model outputs logits; logits are summed and then argmaxed.
    """
    final_preds = []

    #Loop through each batch
    for batch in test_loader:
        
        #Always ensure we have a Tensor on the correct device
        images = batch if isinstance(batch, torch.Tensor) \
               else torch.as_tensor(batch)

        
        logits_sum = 0

        #inference only, no grads
        with torch.no_grad():

            #loop through every model
            for mod in models:
                
                logits_sum += mod(images)

            preds = logits_sum.argmax(dim=1) #argmax for classes

        #Always stored as a tensor
        #detach: remove graph references (Bug fix by Chat GPT)
        #clone: copy just to make sure things are okay
        
        final_preds.append(preds.cpu().detach().clone())

    #Concatenate all batches + convert to numpy arr
    final_preds = torch.cat(final_preds, dim=0).numpy()
    
    return final_preds #np arr final predictions


#ChatGPT Plots
def plot_training_curves(metric, title="Training Curves"):
    epochs = range(1, len(metric["train_loss"]) + 1)
    stop_epoch = metric.get("stop_epoch", None)

    plt.figure(figsize=(12,5))

    # --- Loss Plot ---
    plt.subplot(1,2,1)
    plt.plot(epochs, metric["train_loss"], label="Train Loss")
    plt.plot(epochs, metric["val_loss"], label="Val Loss")

    if stop_epoch is not None:
        plt.axvline(stop_epoch, color='red', linestyle='--',
                    label=f"Stop epoch: {stop_epoch}")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves ResNet18")
    plt.legend()

    # --- Accuracy Plot ---
    plt.subplot(1,2,2)
    plt.plot(epochs, metric["train_acc"], label="Train Acc")
    plt.plot(epochs, metric["val_acc"], label="Val Acc")

    if stop_epoch is not None:
        plt.axvline(stop_epoch, color='red', linestyle='--',
                    label=f"Stop epoch: {stop_epoch}")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

#Chat GPT Generated Conf Matrix
def print_final_report(preds, labels):
    
    preds_np = preds.numpy()
    labels_np = labels.numpy()

    conf_mat = confusion_matrix(labels_np, preds_np)

    print("\nClassification Report:")
    print(classification_report(labels_np, preds_np))

    #print(confusion_matrix(labels_np, preds_np))
    plt.figure(figsize=(8,6))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix for ResNet18")
    plt.show()
