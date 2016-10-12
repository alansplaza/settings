import numpy as np
import matplotlib.pyplot as plt

N = 100 # number of per class
D = 2 # dimensionality
K = 3 # number of classes

X = np.zeros((N*K, D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels

for j in xrange(K):
    ix = range(N*j, N*(j+1))
    r = np.linspace(0.0, 1, N) # radius
    t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2 # theta
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j

plt.scatter(X[:,0], X[:,1], c=y, s=40, cmap=plt.cm.Spectral) # c: color, s: size


#%%
# Train a Softmax Linear Classifier

# initialize parameters randomly
W = 0.01 * np.random.randn(D, K) # 2*3
b = np.zeros((1, K))

# compute class scores for a linear classfier
scores = np.dot(X, W) + b

print scores
#%%
num_examples = N * K

# hypermeters
reg = 0.01

# get unnormalized probabilities
exp_scores = np.exp(scores)

# normalize them for each sample
probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

correct_logprobs = -np.log(probs[range(num_examples), y])

# compute the loss: average cross-entropy loss and regularizations
data_loss = np.sum(correct_logprobs) / num_examples
reg_loss = 0.5 * reg * np.sum(W*W)

loss = data_loss + reg_loss

print loss
