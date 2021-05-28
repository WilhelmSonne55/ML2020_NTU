import numpy as np
import sys

np.random.seed(0)
X_train_fpath = sys.argv[1]
Y_train_fpath = sys.argv[2]
X_test_fpath = sys.argv[3]
output_fpath = sys.argv[4]

# Parse csv files to numpy array
with open(X_train_fpath) as f:
    next(f)
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
with open(Y_train_fpath) as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype = float)
with open(X_test_fpath) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)

def _normalize(X, train = True, specified_column = None, X_mean = None, X_std = None):
    # This function normalizes specific columns of X.
    # The mean and standard variance of training data will be reused when processing testing data.
    #
    # Arguments:
    #     X: data to be processed
    #     train: 'True' when processing training data, 'False' for testing data
    #     specific_column: indexes of the columns that will be normalized. If 'None', all columns
    #         will be normalized.
    #     X_mean: mean value of training data, used when train = 'False'
    #     X_std: standard deviation of training data, used when train = 'False'
    # Outputs:
    #     X: normalized data
    #     X_mean: computed mean value of training data
    #     X_std: computed standard deviation of training data

    if specified_column == None:
        specified_column = np.arange(X.shape[1])
    if train:
        X_mean = np.mean(X[:, specified_column] ,0).reshape(1, -1)
        X_std  = np.std(X[:, specified_column], 0).reshape(1, -1)

    X[:,specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)
     
    return X, X_mean, X_std

def _train_dev_split(X, Y, dev_ratio = 0.25):
    # This function spilts data into training set and development set.
    train_size = int(len(X) * (1 - dev_ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]

def _shuffle(X, Y):
    # This function shuffles two equal-length list/array, X and Y, together.
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def _sigmoid(z):
    # Sigmoid function can be used to calculate probability.
    # To avoid overflow, minimum/maximum output value is set.
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))

def _f(X, w, b):
    # This is the logistic regression function, parameterized by w and b
    #
    # Arguements:
    #     X: input data, shape = [batch_size, data_dimension]
    #     w: weight vector, shape = [data_dimension, ]
    #     b: bias, scalar
    # Output:
    #     predicted probability of each row of X being positively labeled, shape = [batch_size, ]
    return _sigmoid(np.matmul(X, w) + b)

def _predict(X, w, b):
    # This function returns a truth value prediction for each row of X 
    # by rounding the result of logistic regression function.
    return np.round(_f(X, w, b)).astype(np.int)
    
def _accuracy(Y_pred, Y_label):
    # This function calculates prediction accuracy
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc
	
def _cross_entropy_loss(y_pred, Y_label):
    # This function computes the cross entropy.
    #
    # Arguements:
    #     y_pred: probabilistic predictions, float vector
    #     Y_label: ground truth labels, bool vector
    # Output:
    #     cross entropy, scalar
    cross_entropy = -np.dot(Y_label, np.log(y_pred)) - np.dot((1 - Y_label), np.log(1 - y_pred))
    return cross_entropy

def _gradient(X, Y_label, w, b):
    # This function computes the gradient of cross entropy loss with respect to weight w and bias b.
    y_pred = _f(X, w, b)
    pred_error = Y_label - y_pred
    w_grad = -np.sum(pred_error * X.T, 1)
    b_grad = -np.sum(pred_error)
    return w_grad, b_grad

def train(x_train, y_train, x_val, y_val, lamb = 0, p = False):
  # Zero initialization for weights ans bias
  w = np.zeros((x_train.shape[1],)) 
  b = np.zeros((1,))
  train_size = x_train.shape[0]
  val_size = x_val.shape[0]

  # Some parameters for training    
  max_iter = 100
  batch_size = 8
  learning_rate = 0.5

  # Keep the loss and accuracy at every iteration for plotting
  train_loss_data = []
  val_loss_data = []
  train_acc_data = []
  val_acc_data = []

  # Calcuate the number of parameter updates
  step = 1

  # Iterative training
  for epoch in range(max_iter):
      # Random shuffle at the begging of each epoch
      x_train, y_train = _shuffle(x_train, y_train)
          
      # Mini-batch training
      for idx in range(int(np.floor(train_size / batch_size))):
          x = x_train[idx*batch_size:(idx+1)*batch_size]
          y = y_train[idx*batch_size:(idx+1)*batch_size]

          # Compute the gradient
          w_grad, b_grad = _gradient(x, y, w, b)
          w_grad = w_grad + lamb*w

          # gradient descent update
          # learning rate decay with time
          w = w - learning_rate/np.sqrt(step) * w_grad / batch_size
          b = b - learning_rate/np.sqrt(step) * b_grad / batch_size

          step = step + 1
              
      # Compute loss and accuracy of training set and development set
      #training part
      y_train_pred = _f(x_train, w, b)
      y_train_pred_round = np.round(y_train_pred)

      train_accuracy = _accuracy(y_train_pred_round, y_train)
      train_acc_data.append(train_accuracy)

      train_loss = _cross_entropy_loss(y_train_pred, y_train) / train_size + (lamb*np.dot(w,w.T))/(2*train_size)
      train_loss_data.append(train_loss)

      #validation part
      y_val_pred = _f(x_val, w, b)
      y_val_pred_round = np.round(y_val_pred)

      val_accuracy = _accuracy(y_val_pred_round, y_val)
      val_acc_data.append(val_accuracy)

      val_loss = _cross_entropy_loss(y_val_pred, y_val) / val_size + (lamb*np.dot(w,w.T))/(2*val_size)
      val_loss_data.append(val_loss)
  if p == True:
    print('training loss:', train_loss)
    print('training prediction:', train_accuracy)
    print('validation loss:', val_loss)
    print('validation prediction:', val_accuracy)

  return train_loss, train_loss_data, train_acc_data, val_loss, val_loss_data, val_acc_data, w, b

def save(X_test, w, b):
  predictions = _predict(X_test, w, b)
  with open(output_fpath.format('logistic'), 'w') as f:
      f.write('id,label\n')
      for i, label in  enumerate(predictions):
          f.write('{},{}\n'.format(i, label))
  
# Normalize training and testing data
X_train, X_mean, X_std = _normalize(X_train, train = True)
X_test, _, _= _normalize(X_test, train = False, specified_column = None, X_mean = X_mean, X_std = X_std)
    
# Split data into training set and development set
dev_ratio = 0.1
X_train, Y_train, X_dev, Y_dev = _train_dev_split(X_train, Y_train, dev_ratio = dev_ratio)


index = [210, 507, 212,   0, 358, 213, 113, 191, 192, 116, 211, 120, 117,
       114,  98, 217, 175, 124, 119, 125, 122, 161, 115, 360, 173,   2,
        73,  94, 103,  86, 317, 297, 165,   3, 218, 367, 111, 320, 136,
         9,  63,  13, 500, 106, 450,  68, 300, 315, 168, 216, 104,  33,
       359,  69, 171,  19, 140, 391, 505, 123,  89, 163,  82, 187, 429,
        80,  41, 128, 109, 468, 322,  36, 384, 458, 420, 131, 294, 469,
       388, 504,  74, 386, 203, 389, 312,   8, 200, 484, 467, 445, 155,
       428, 409, 463, 365, 381, 366, 368,  96, 373, 329,  93,  30,  44,
       158, 374,  35, 132, 341,  31,  38, 105, 377, 180, 474, 242,  97,
       396, 321, 190, 309, 433,  53, 222, 249, 135,  87, 151,  45, 436,
       344, 186, 277, 408,  67, 157,  75, 205, 110, 181, 316, 432,  71,
       415,  84, 147,  21, 282, 214,   7, 486, 178,  42, 138,  27, 325,
        24, 480, 424, 170,  57, 385, 287,  64, 399, 394, 442, 470, 208,
       184, 247, 488, 311, 490, 265, 335,  56, 201, 281, 137,  50, 326,
       204, 275,   6, 139,  32, 406, 241,  23, 235, 498, 164,  85, 459,
       127, 259, 248, 199, 185, 227, 423, 263, 465, 229,  22, 145, 179,
       412, 129, 246,  28,  26, 143, 387, 457, 273, 395,  62,  14, 224,
       472, 462,  11,  46, 402,  29, 482, 141, 144, 251, 496, 382, 441,
       130,  92, 487,  78, 209, 196, 301, 134, 243, 237, 238, 330, 503,
       333, 342, 118, 473, 244, 435, 133, 419, 489, 451, 220, 397, 177,
         4, 195, 452,  15, 493,  12, 268, 274, 194, 245, 324, 400, 506,
       501,  20, 146,  90,  99, 215, 202, 308, 453, 357,  37, 446, 219,
       279, 225, 497, 401, 478, 167, 108,  60, 160, 174,   1, 414, 230,
       305,  18, 152, 271, 121, 418, 370, 153, 481,  52, 101, 296,  58,
       460, 369, 126,  39,  59, 149, 440,  54, 232, 159, 183, 448, 112,
       306, 198, 353, 221, 255, 454,  43, 150, 295,  40, 142,  55, 188,
        91, 354, 492, 228, 337, 327, 352, 346, 476, 398, 464, 403, 197,
       156,  47, 471, 430, 253, 272,  76, 405, 260, 107,  83, 336, 426,
       262, 372, 499, 427, 348, 479, 233, 193, 299, 291,  79, 257, 345,
       485, 176,  10, 404, 303, 319, 444,  49,  77,  16, 411, 421, 266,
         5, 226, 166,  95, 254, 270, 380, 318,  88, 383,  70, 332, 413,
       258, 261, 231, 437, 461, 483,  34, 466, 298, 338, 347, 393, 407,
       447, 269, 207, 417, 439, 495,  81, 477, 362,  48, 154, 350, 355,
       340, 331, 509, 508, 162, 351, 379, 223, 349, 339, 410, 239, 494,
       438, 416, 356, 280, 328,  61, 288,  72, 449, 491,  66, 456, 102,
       304, 434, 364, 502,  65, 278,  51, 100, 148, 169, 182,  17, 276,
       392, 240, 283, 443, 375, 323, 334, 343, 425, 371, 292, 285, 422,
       206, 431, 314, 267, 252, 455, 264, 236, 189,  25, 361, 475, 378,
       256, 363, 376, 289, 172, 234, 313, 250, 290, 307, 284, 390, 310,
       293, 286, 302]

ind = index[:170]
train_loss, train_loss_data, train_acc_data, val_loss, val_loss_data, val_acc_data, w, b \
= train(X_train[:,ind], Y_train, X_dev[:,ind], Y_dev, 0.0025, True)

save(X_test[:,ind], w, b)