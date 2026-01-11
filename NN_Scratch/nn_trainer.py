import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from NeuralNet import NeuralNet, LabelSmoothingLoss
from nn_utils import flip_horizontal, compute_sampling_probs


def train_nn_model(seed, data, hyperparams):
    """
    Train a single NN given a specific seed.
    
       - batch sampling (with class balancing).
       - augmentation (horizontal flip).
       - validation loss computation + classif report + conf matrix USING seaborn (sns.heatmap)
    """

    print(f"\nTraining NN with seed {seed}")

    #Set Seed
    np.random.seed(seed)

    #Retrieve params from data dict
    images= data["images"]
    labels = data["labels"]
    scaler = data["scaler"]
    train_idx= data["train_idx"]
    X_val_scaled = data["X_val_scaled"]
    y_val = data["y_val"]

    #Retrieve hyperparameters
    input_dim = hyperparams["input_dim"]
    num_classes = hyperparams["num_classes"]
    batch_size = hyperparams['batch_size']
    epochs = hyperparams['epochs']
    lr_initial = hyperparams["lr_initial"]
    weight_decay = hyperparams['weight_decay']
    momentum= hyperparams["momentum"]
    dropout_prob= hyperparams["dropout_prob"]

    # Shuffle training indices using seed
    local_idx = train_idx.copy()
    np.random.shuffle(local_idx)

    X_train = images[local_idx]
    y_train = labels[local_idx]

    #Compute sampling probabilities (tries to balance out more classes)
    samp_probs = compute_sampling_probs(y_train, num_classes)

    #init model
    model = NeuralNet(input_dim, num_classes, dropout_prob, weight_decay)

    #define validation loss
    val_loss_func = LabelSmoothingLoss(0.1, num_classes)

    train_losses = []
    val_losses = []

    lr = lr_initial  # learning rate that will decay later

    #Training Loop
    for ep in range(epochs):

        #drop LR by × 0.1 at epoch 20
        if ep == 20:
            lr *= 0.1
            print(f"Learning rate decayed to {lr}")

        batch_losses = []
        n_batches = int(np.ceil(len(X_train) / batch_size))

        #Train for all batches
        #Transform images

        count = 0
        for batch in range(n_batches):

            #Sample balanced batches.
            batch_ids = np.random.choice(len(X_train), batch_size, p=samp_probs)

            #Extract images + augment
            X_batch = X_train[batch_ids].astype(np.float32) / 255.0
            X_batch = flip_horizontal(X_batch)

            #Flatten + normalize using scaler implemented from scratch
            Xb_flat = X_batch.reshape(batch_size, -1)
            Xb_scaled = scaler.normalize(Xb_flat)

            y_batch = y_train[batch_ids]

            #Single train step (forward + backward + update)
            loss = model.backward(Xb_scaled, y_batch, lr , momentum,)
            batch_losses.append(loss)

                

        #Average loss every epoch
        mean_train_loss = float(np.mean(batch_losses))
        train_losses.append(mean_train_loss)

        #Validation Loss
        with np.errstate(over="ignore"):  # avoids warnings on exp()
            
            logits_val = model.forward(X_val_scaled, training=False)
            val_loss = val_loss_func.forward(logits_val, y_val)

        val_losses.append(float(val_loss))

        # Print progress every 10 epochs
        if (ep % 10 == 0) or (ep == epochs - 1):
            print(f"Epoch {ep}: train loss: {mean_train_loss:.4f} | val loss: {val_loss:.4f}" )

    return model, train_losses, val_losses
#End of train model function

#Plot functions
#Adapted from chatGPT
#Use only numpy and matplotlib
#Link Reference: https://chatgpt.com/share/693bad99-5f14-8002-997f-c60b4521d134

def plot_loss_curves(train_losses, val_losses, seed):
    epochs = np.arange(len(train_losses))
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss curves (seed {seed})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def confusion_matrix_np(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

#Adapted from ChatGPT 
def classification_report_np(y_true, y_pred, num_classes):
    cm = confusion_matrix_np(y_true, y_pred, num_classes)
    total = cm.sum()
    acc = np.trace(cm) / total if total > 0 else 0.0

    print("\n --Validation Results--")
    print(f"Validation Accuracy: {acc:.4f}\n")
    print(f"{'Cls':<4} {'Prec':<8} {'Rec':<8} {'F1':<8} {'Support':<8}")
    print("-" * 40)

    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        sup = cm[c, :].sum()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        print(f"{c:<4} {prec:.4f}    {rec:.4f}    {f1:.4f}    {sup:<8}")
    return cm

def plot_confusion_matrix_heatmap(cm, class_names=None, title=None):
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    plt.figure(figsize=(5, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.show()
