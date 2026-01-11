import numpy as np

#References:
#1. Adrej Karpathy's NN: Zero to Here Playlist (especially videos 1,4(Activations,Gradients,BatchNorm) and 5(Backprop Ninja))
#2. Michael Nielsen's Book and Github for Backprop and NN implementation from scatch
#3. Andrew Ng's Hyperparam Tuning Playlist (especially Momentum, Dropout,BatchNorm)

#Links:
#1. Karpathy: https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ
#2. Nielsen's Github Repo: https://github.com/mnielsen/neural-networks-and-deep-learning
#3. Andrew Ng: https://www.youtube.com/playlist?list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc

#Helper functions

def py_init(n_in, n_out):
    """
    Uses He/Kaiming's Weight initialization for ReLU layers.
    variance = 2/inputs
    doesn't solve but helps reduce vanishing/exploding gradients (prevents stuff from blowing out)

    Andrew Ng: set variance to 2/inputs for ReLU -> Can tune the variance as a hyperparam  (can help but not very important)
    
    Link: https://www.youtube.com/watch?v=s2coXdufOzE&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=11
    """
    #std dev from the formula sqrt(2 / fan_in)
    std_dev = np.sqrt(2.0 / n_in)

    #weights from normal distribution scaled by std dev
    W = np.random.randn(n_in, n_out).astype(np.float32) * std_dev

    #by covention: biases init to 0
    b = np.zeros((1, n_out), dtype=np.float32)

    return W, b


def leaky_relu(x, alpha=0.01):
    """
    small gradient wgen grad is -ve instead of 0
    prevents neurons from completely being shut off.
    Can accelerate learning
    """
    return np.maximum(alpha * x, x)


def leaky_relu_backward(grad, x, alpha=0.01):
    """
    derivative is 1 when x >= 0, and alpha when x < 0.
    """
    #grad is 1 except where x < 0 then slope = alpha
    mask = np.ones_like(x)
    mask[x < 0] = alpha

    #apply chain rule
    return grad * mask


class Linear:
    """
    Fully connected layer (W.T x + b).
    Momentum smoothes out the weights of gradient descent (Andre Ng's Specialization)
    Oscillations in the vertical direction decrease and become smaller/smoother (Andrew)
    Beta usually 0.9 (average over last 10 iter gradients)
    
    """

    def __init__(self, n_in, n_out):
        
        # initialize params using pytorch's init
        self.W, self.b = py_init(n_in, n_out)

        #init momentum vectors with 0 (Andrew Ng)
        #zeros like used to prevent specifying shape -> np.zeros directly got me a lot of bugs       
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)

        # cache for backprop
        self.input_cache = None     #x
        self.output_cache = None    #z = W.T x + b

    def forward(self, x):

        #z = W.T x + b
        #save x for backprop later

        self.input_cache = x
        self.output_cache = np.dot(x, self.W) + self.b
        return self.output_cache

    def backward(self, grad, weight_decay):
        
        #grad: gradient from the next layer (shape: N, n_out)

        #dW = x.T * grad + weight_decay * W
        #db = sum of grad_out over samples
        #dx = grad * W.T
        
        dW = np.dot(self.input_cache.T, grad) + (weight_decay * self.W)

        db = grad.sum(axis=0, keepdims=True)

        # gradient wrt input
        dx = np.dot(grad, self.W.T)

        return dx, dW, db

    def update(self, dW, db, lr, momentum):
        
        #SGD + momentum
        #velocity terms: vW , vb
        #v = m * v + grad
        
        # update momentum vectors
        self.vW = (momentum * self.vW) + dW
        self.vb = (momentum * self.vb) + db

        # update params
        self.W -= (lr * self.vW)
        self.b -= (lr * self.vb)

class Dropout:

    """
    Inverted Dropout (Andrew Ng)
    Multiply 0 or 1 by vector: this helps zero out a fracton of the neurons.
    Turn off neurons randomly to prevent network from relying too much on specific ones.
    Smaller NN should have a regularizing effect
    Reference Link: https://www.youtube.com/watch?v=D8PJAL-MZv8&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=6
    """

    def __init__(self, prob):
        
        self.p = prob # prop between 0 and 1 -> drop p % neurons
        
        #mask stores neurons that do not die: will be used for backprop later. grads computed for live neurons only
        self.mask = None

    def forward(self, x, training):
        
        #If no dropout, do nothing
        if (not training) or (self.p <= 0.0):
            return x

        #dropout mask: True = keep neuron
        # rand has range [0, 1). rand > p means keep_prob = 1 - p (Anrew Ng's video)
        
        self.mask = (np.random.rand(*x.shape) > self.p).astype(np.float32)

        #Inverted dropout: scale live neurons by (1-p)
        #Ensures that expected value of activations remains the same after dropping some neurons
        self.mask /= (1.0 - self.p) 

        result = self.mask * x
        
        return result

    def backward(self, grad):
        #apply mask : because gradient only computed for neurons that were not dead during forward pass
        
        result = self.mask * grad
        return result

class BatchNorm:
    """
    (Andrew Ng): Normalize not just input layer but activations of every layer
    Normalization help make the countours more symmetrical -> helps accelerate training
    We do that for every layer in BatchNorm.
    Technically normalize z, not a. Done much more often according to Andrew Ng

    BatchNorm reduces covariate shift. 

    Adapted ChatGPT's code to implement BatchNorm: https://chatgpt.com/share/693b824d-cd6c-8002-a8fa-3ab0cd17f562

    ** DIFFERENT BETA THAN MOMENTUM

    Gamma: scale param, Beta: shift param
    """

    def __init__(self, dim, momentum=0.9):
        
        #Gamma and beta are learnable params for our moel (Andrew Ng)
        #Use grad descent to update just like for weights
        #Gamme allows us to set mean to whatever we want as we do not necessarily want our hidden unit values to have mean 0 and st_dev 1.
        #Can have our own range of vals for mean and var controlled by 2 explicit params. Allows normalization of hidden units.

        self.gamma = np.ones((1, dim), dtype=np.float32)
        self.beta = np.zeros((1, dim), dtype=np.float32)

        #Running mean and var used for evaluation
        self.running_mean = np.zeros((1, dim), dtype=np.float32)
        self.running_var = np.ones((1, dim), dtype=np.float32)

        self.momentum = momentum

        # Saved values for backward pass
        self.cache = None        #(x, mean, var)
        self.norm_cache = None   #normalized_x

    #forward pass
    #output: gamma * normalized_x + beta : scale and shift
    def forward(self, x, training=True):
        """
        For training:
            - compute batch mean/variance
            - normalize input
            - update running stats

        For eval:
            - use running mean/variance for normalization
        """

        if training:
            #feature-wise
            mean = x.mean(axis=0, keepdims=True)
            var = x.var(axis=0, keepdims=True)

            #Update running mean and var (used later for inference).
            self.running_mean = (self.momentum * self.running_mean + ((1 - self.momentum) * mean))
            self.running_var = (self.momentum * self.running_var + (1 - self.momentum) * var)

            #Normalize input...ad epsilon in case std turns out to be zero
            norm = (x - mean) / np.sqrt(var + 1e-5)

            #Save for backprop
            self.cache = (x, mean, var)

        else:
            # In inference mode, we use the accumulated running stats
            norm = (x - self.running_mean) / np.sqrt(self.running_var + 1e-5)

        #Save normalized vals: needed for dgamma
        self.norm_cache = norm

        #Scale (gamma) and shift (beta)
        return self.gamma * norm + self.beta

    #backward pass
    def backward(self, grad):
        """
        Returns:
            dx - grad wrt input x
            dgamma, ddbetta
        """

        x, mean, var = self.cache
        N = x.shape[0]

        #Gradient of scale Gamma and shift Beta params
        dgamma = np.sum(grad * self.norm_cache, axis=0, keepdims=True)
        dbeta = grad.sum(axis=0, keepdims=True)

        #Useful values (chatGPT fixed a bug I had here)
        #epsilon in case var = 0.
        
        inv_std = 1.0 / np.sqrt(var + 1e-5)
        dx_hat = grad * self.gamma  # gradient w.r.t normalized x

        #ChatGPT Reference
        dvar = np.sum(dx_hat * (x - mean) * -0.5 * inv_std**3,
                      axis=0, keepdims=True)

        dmean = np.sum(dx_hat * -inv_std, axis=0, keepdims=True)
        dmean += dvar * np.mean(-2.0 * (x - mean), axis=0, keepdims=True)

        #input gradient: chain rule
        dx = dx_hat * inv_std
        dx += (2.0 / N) * dvar * (x - mean)
        dx += dmean / N

        return dx, dgamma, dbeta

    #Gradient Descent for beta and gamma
    #Vanilla SGD
    def update(self, dgamma, dbeta, lr):
        
        self.gamma -= lr * dgamma
        self.beta -= lr * dbeta

        
class LabelSmoothingLoss:
    """
    Cross-entropy loss + smoothing.
    Instead of using a hard labels, soften it a bit so that model doesn't become overconfident.
    Stabilizes it and can help with better generalization
    Form of regularizer
    """

    def __init__(self, smoothing=0.1, num_classes=5):
        
        # smoothing = epsilon
        self.smoothing = smoothing
        self.num_classes = num_classes
        
        #Store forward vals for backprop later
        self.probs = None
        self.targets = None

    def forward(self, logits, y):
        
        """
        logits with shape (N x C)

        1. Softmax to turn raw output logits to probabilities
        2. Smooth labels
        3. Cross entropy with smoothed labels
        """

        #Softmax
        #subtract max per row to improve numerical stability (GPT Reference)
        
        max_vals = np.max(logits, axis=1, keepdims=True)
        exp_vals = np.exp(logits - max_vals)
        probs = exp_vals / exp_vals.sum(axis=1, keepdims=True)
        
        self.probs = probs   # softmax output: save for backprop

        N = len(y)

        #one-hot encoding
        targets = np.zeros_like(probs)
        targets[np.arange(N), y] = 1.0

        #label smoothing
        eps = self.smoothing
        
        #soften distribution
        smooth_targets = (targets * (1 - eps)) + (eps / self.num_classes)
        self.targets = smooth_targets

        #Smooth Cross-entropy
        #small delta val to avoid log 0.
        loss = -np.sum(smooth_targets * np.log(probs + 1e-9)) / N

        return loss

    def backward(self):

        #Return gradient wrt to raw outputs

        #grad = (probs - smoothed_target) / N
        return (self.probs - self.targets) / len(self.probs)

