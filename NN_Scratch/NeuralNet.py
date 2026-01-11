import numpy as np

from nn_layer import Linear, BatchNorm, Dropout, LabelSmoothingLoss, leaky_relu, leaky_relu_backward

class NeuralNet:
    """
    Neural Network / Fully-connected MLP
    Architecture: [Linear -> BatchNorm -> LeakyReLU -> Dropout] x 3 -> Linear (5 classes)
    Architectural Pattern Inspired AND ADAPTED by Nielsen's Code Structure
    Reference Link: https://github.com/mnielsen/neural-networks-and-deep-learning
    """
    
    def __init__(self, input_dim, num_classes, dropout_prob, weight_decay):
        self.weight_decay = weight_decay
        self.layers = []

        #Can change architecture dynamically
        
        self.hidden_sizes = [256, 128, 64] 

        #self.hidden_sizes = [512, 256, 128, 64]
        #self.hidden_sizes = [1024, 512, 256, 128, 64] 
        
        self.sizes = [input_dim] + self.hidden_sizes + [num_classes]

        #Num linear layers
        self.n_layers = len(self.sizes) - 1

        #init layers
        #Build sequence of layers.
        #Each layer: 4 components (linear,batch,relu,dropout)
        
        for i in range(self.n_layers):
            
            #Linear layer for all layers
            #input size = size of currLayer, output size = size of next layer
            self.layers.append(Linear(self.sizes[i], self.sizes[i+1]))
            
            #BatchNorm & Dropout only for first N-1 layers
            if i < (self.n_layers-1):
                self.layers.append(BatchNorm(self.sizes[i+1]))
                self.layers.append(Dropout(dropout_prob))
        
        #last layer is output layer, hence -1 index
        self.output_layer = self.layers[-1] 

        #Cross Entropy + Smoothing
        self.loss_func = LabelSmoothingLoss(0.1, num_classes)


    #Forward Pass
    def forward(self, X, training=True):
        out = X

        # Hidden: (Linear -> BN -> ReLU -> Dropout)      
        for i in range( len(self.hidden_sizes) ):
            offset = i * 3

            linear = self.layers[offset]
            bnorm = self.layers[offset + 1]
            drop = self.layers[offset + 2]

            #forward flow
            z = linear.forward(out)
            z_bnorm = bnorm.forward(z, training=training)
            relu = leaky_relu(z_bnorm)
            out = drop.forward(relu, training=training)

        # Final output layer (Linear only)
        logits = self.output_layer.forward(out)
        return logits

    def backward(self, X, y, lr, momentum):
        """
        1 training step has:
        
          Forward pass
          Loss function
          Backprop output
          Backprop through all hidden layers (reverse order)
          Update params
        """
    
        #Forward and Loss Function
        logits = self.forward(X, training=True)
        loss = self.loss_func.forward(logits, y)
    
        #Logits' Gradient
        grad = self.loss_func.backward()
    
        #Backprop for last Linear layer
        grad, dW_out, db_out = self.output_layer.backward(grad, self.weight_decay)
        self.output_layer.update(dW_out, db_out, lr, momentum)

        #Backprop for hidden layers (in reverse order)

        #Hideen layer:   [ Linear_i, BN_i, Drop_i, Linear_(i+1), BN_(i+1), etc... ]
        #Each block is 3 layers: Linear, BNorm, Dropout
        #iterate from last hidden block back to first.
        
        for i in reversed(range(len(self.hidden_sizes))):
            
            offset = i * 3
    
            linear = self.layers[offset]
            bnorm     = self.layers[offset + 1]
            drop   = self.layers[offset + 2]
    
            #Dropout backprop
            grad = drop.backward(grad)
    
            #ReLU backprop
            #Recompute BNorm output using EVAL mode to get activation mask (Adapted from chatGPT -> fixed bug)
            
            z = linear.output_cache
            z_bnorm_eval = bnorm.forward(z, training=False)
            grad = leaky_relu_backward(grad, z_bnorm_eval)
    
            #BatchNorm backprop
            grad, dgamma, dbeta = bnorm.backward(grad)
            bnorm.update(dgamma, dbeta, lr)
    
            # Linear Backprop
            grad, dW, db = linear.backward(grad, self.weight_decay)
            linear.update(dW, db, lr, momentum)
    
        return loss