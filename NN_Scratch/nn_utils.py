import numpy as np
import pickle
import matplotlib.pyplot as plt

#Standard scaler From Scratch
#Stratified train_test_split of sklearn
#Horizontal flipping of images 
#Test Time Augmentation for predictions (Horizontal Flip + original averaged)

#Used Gen AI To UNERSTAND the logic

#Adapted the code:
#Links:
#https://chatgpt.com/share/693b4ad1-3b54-8002-8db3-00a9de92d2a4
#https://chatgpt.com/share/693b5152-bcec-8002-b827-32605691a011

class Scaler_Scratch:

    def normalize_train(self, X):

        #Use float 32 for NN instead of standard float 64
        mean_train = X.mean(axis=0, keepdims=True).astype(np.float32)
        std_dev_train = X.std(axis=0, keepdims=True).astype(np.float32)

        #(GPT code not readable, i prefer simpler readable code)
        #std_dev[std_dev == 0] = 1.0 

        if np.all(std_dev_train == 0):
            std_dev_train = 1.0
        
        self.mean = mean_train
        self.scale = std_dev_train
        
        return self

    def normalize(self, X):
        return (X - self.mean) / self.scale

#End of Scaler class

#The other functions are utility functions used in notebook

#Randomly flip 50% of images horitzontally
#Increases robustness
def flip_horizontal(batch):
    n = len(batch)

    #50% chance of getting flipped
    flip_mask = (np.random.rand(n) < 0.5)

    #copy to not overwrite batch
    augmented_batch = batch.copy()

    for i in np.where(flip_mask)[0]:
        augmented_batch[i] = np.flip(batch[i], axis=1)   #axis=1: horizontal flip    
    
    return augmented_batch
    

#Test Time Augmentation + Softmax average
def predict_tta(model, X_scaled, scaler, height, width, channels):
    """
    Test-Time Augmentation using a single horizontal flip.
    
    This function:
        1. Predicts on the original scaled input.
        2. Unscales the input back to pixel space.
        3. Horizontally flips each image.
        4. Re-scales the flipped images.
        5. Predicts on the flipped version.
        6. Averages both probability vectors.
    
    NOTE: Functionality and numerical behavior are preserved exactly.
    """
    #Predict original inputs
    weights_original = model.forward(X_scaled, training=False)

    #Softmax for original images
    max_original = np.max(weights_original, axis=1, keepdims=True)
    exp_original = np.exp(weights_original - max_original)
    probs_original = exp_original / exp_original.sum(axis=1, keepdims=True)
    
    #Flip input horizontally
    # Undo scaling: return to pixel intensity space
    X_unscaled = (X_scaled * scaler.scale) + scaler.mean

    #Convert vectors back into tensors
    images = X_unscaled.reshape(-1, height, width, channels)

    # Horizontal flip along width dimension
    #Adapted from GPT
    flipped_images = images[:, :, ::-1, :]

    #Flatten flipped images (N x D)
    flipped_flattened = flipped_images.reshape(len(flipped_images), -1)

    #Restandardize for preprocessing
    flipped_scaled = scaler.normalize(flipped_flattened)

    #Predict flipped images
    weights_flipped = model.forward(flipped_scaled, training=False)

    #Softmax for flipped images
    max_flipped = np.max(weights_flipped, axis=1, keepdims=True)
    exp_flipped = np.exp(weights_flipped - max_flipped)
    probs_flipped = exp_flipped / exp_flipped.sum(axis=1, keepdims=True)

    #Return average of original and flipped predictions
    return (probs_original + probs_flipped) / 2.0

#Train and Test Data Functions to Load Data from pickle
def load_train_data(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
        
    images = data["images"]
    labels = np.array(data["labels"]).reshape(-1)

    return images, labels


def load_test_data(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    images = data["images"]
    return images

#Shuffle Indexes and Return them for train-test split
#Adapted from ChatGPT code: https://chatgpt.com/share/693b4ad1-3b54-8002-8db3-00a9de92d2a4

def stratified_train_test_indices(labels, split_ratio=0.15, num_classes=5, split_seed=999):
    """
        Gather indices for each class + shuffle them
        Take first ratio % as validation
        (Same logic used by sklearn's split according to GPT)
    """
    rand_generator = np.random.RandomState(split_seed)

    indices = np.arange(len(labels))

    train_indices, val_indices = [], []

    #Loop and split by class
    for label in range(num_classes):
        
        #all indices belonging to class
        class_id = indices[labels == label]

        #Shuffle
        rand_generator.shuffle(class_id)

        #samples in valid set
        n_val = int(len(class_id) * split_ratio)

        #first n goes to validation, rest goes to train set
        val_indices.extend(class_id[:n_val])
        train_indices.extend(class_id[n_val:])

    # After combining all classes, shuffle again so classes aren't grouped
    rand_generator.shuffle(train_indices)
    rand_generator.shuffle(val_indices)

    return np.array(train_indices), np.array(val_indices)


#Balanced sampling used in trainer.py
def compute_sampling_probs(labels, num_classes):
    """
    Inverse weights: rare classes get higher weights during sampling
    """
    counts = np.bincount(labels, minlength=num_classes) #count num samples per classes
    weights = 1.0 / (np.sqrt(counts) + 1e-6) #inverse frequency (epsilon to avoid zero division)
    
    w = weights[labels] #Normalize weights
    
    return w / w.sum()