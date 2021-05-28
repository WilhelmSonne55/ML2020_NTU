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

def calAccuracy(X_train, Y_train):
  # Compute in-class mean
  X_train_0 = np.array([x for x, y in zip(X_train, Y_train) if y == 0])
  X_train_1 = np.array([x for x, y in zip(X_train, Y_train) if y == 1])

  mean_0 = np.mean(X_train_0, axis = 0)
  mean_1 = np.mean(X_train_1, axis = 0)  

  # Compute in-class covariance
  feature_size = X_train.shape[1]
  cov_0 = np.zeros((feature_size, feature_size))
  cov_1 = np.zeros((feature_size, feature_size))

  for x in X_train_0:
      cov_0 += np.dot(np.transpose([x - mean_0]), [x - mean_0]) / X_train_0.shape[0]
  for x in X_train_1:
      cov_1 += np.dot(np.transpose([x - mean_1]), [x - mean_1]) / X_train_1.shape[0]

  # Shared covariance is taken as a weighted average of individual in-class covariance.
  cov = (cov_0 * X_train_0.shape[0] + cov_1 * X_train_1.shape[0]) / (X_train_0.shape[0] + X_train_1.shape[0])

  u, s, v = np.linalg.svd(cov, full_matrices=False)
  inv = np.matmul(v.T * 1 / s, u.T)

  # Directly compute weights and bias
  w = np.dot(inv, mean_0 - mean_1)
  b =  (-0.5) * np.dot(mean_0, np.dot(inv, mean_0)) + 0.5 * np.dot(mean_1, np.dot(inv, mean_1))\
      + np.log(float(X_train_0.shape[0]) / X_train_1.shape[0]) 

  # Compute accuracy on training set
  Y_train_pred = 1 - _predict(X_train, w, b)
  print('Training accuracy: {}'.format(_accuracy(Y_train_pred, Y_train)))
  return w, b

def saveCon(X_test, w, b):
  predictions = 1 - _predict(X_test, w, b)
  with open(output_fpath.format('generative'), 'w') as f:
      f.write('id,label\n')
      for i, label in  enumerate(predictions):
          f.write('{},{}\n'.format(i, label))
  
# Normalize training and testing data
X_train, X_mean, X_std = _normalize(X_train, train = True)
X_test, _, _= _normalize(X_test, train = False, specified_column = None, X_mean = X_mean, X_std = X_std)
    
# Split data into training set and development set
dev_ratio = 0.1
X_train, Y_train, X_dev, Y_dev = _train_dev_split(X_train, Y_train, dev_ratio = dev_ratio)


ind2 = [141,  25, 148,  24, 162, 255, 350, 169,  18, 152,  60,  22, 145,
       154, 100, 353,  48,  42, 138, 349,  78, 302,  45,  66, 151, 352,
        67, 165, 174, 286,  95, 346, 200,   8, 414, 327,  19, 153, 140,
       317, 136,  47, 147, 337, 134, 331,  21, 221,  49,   1, 156,  98,
       149, 191, 158, 137, 508, 166,  50, 170,  44,  85,  59, 294, 139,
        32, 131, 339, 192, 223, 429,  52, 504, 108, 493, 340,  74, 168,
       297, 323, 161, 324, 438, 164, 457, 312, 142, 216, 132, 333,  55,
         9, 175, 144, 505, 177,  43, 452, 501, 173,  97, 338, 362,   4,
       472,  99, 359, 150,   3, 172, 356, 205, 481, 127, 130, 213,  68,
       283, 334, 506, 300, 113, 409, 180,  33, 343, 203,  76, 316,  86,
       277, 360,   0, 329,  81,  75,  69, 357, 441, 217, 418,  91, 509,
        77, 440, 423, 193, 218, 507, 133,  34,  71, 195, 497, 128,  93,
       210,  80, 408, 120,   7, 212, 431, 275, 204, 410, 178, 422, 424,
       117, 433, 116, 415, 347, 358,  89, 355, 129, 155, 287,  31, 503,
        94,  82,  40, 419, 211, 451, 371,   2, 342,  27, 449, 215, 171,
       461,  62, 436,  96, 176, 458, 305, 427, 326, 495, 101, 220,  65,
       163, 467,  61, 407,  64, 483, 281,  84, 311, 361,   5, 280, 462,
       465, 190,  70, 450, 411,  87, 239, 476,  36,  13, 123, 470, 320,
       453, 446, 432, 421, 328, 443, 448, 214, 363, 107, 234, 233, 466,
        37, 430, 480, 207,  72, 417,  92, 474, 135, 330, 455, 242, 444,
       303, 439,  11,  35, 479, 122, 240, 261, 179, 292, 491, 475, 187,
       288, 247, 484, 454, 157, 298, 394, 354, 435, 289, 469, 434, 309,
       184, 143, 437, 489, 445,  26, 478, 278, 160,  73, 224,  38, 494,
        83, 199, 209, 428,  90, 473, 366, 231, 257, 252, 487, 420, 426,
       114, 264, 460, 219, 463, 194, 385,  57, 235, 412, 315, 256, 276,
       295, 232, 447, 167, 500, 471, 124, 236, 336,  58, 459, 189, 502,
       425, 325, 492, 335,  46, 188, 183, 243, 296, 241, 416, 197, 225,
       321, 246, 488, 206, 260, 262, 391, 115,   6, 464, 482, 248,  15,
       413, 125, 119,  29, 222, 396, 490,  16, 397, 477, 351,  14, 318,
        28, 237, 271, 268, 159, 104, 238, 229, 274, 468, 498, 103, 253,
       308,  79, 249, 196, 332, 380, 208, 226, 386, 228, 250, 263, 388,
       377, 378, 406, 322, 109, 270, 344, 375, 202, 273, 267, 266, 269,
        88, 372,  39, 379, 201, 181, 403,  54, 106, 198, 373, 486, 301,
        30, 442, 365, 499, 102, 345, 387, 186, 111, 146, 272,  63, 185,
        12, 121,  17, 485, 393, 110, 456, 258, 390,  56, 182, 383, 369,
       374, 395,  20, 392, 381,  51, 384, 404, 291, 364, 254,  53,  41,
       251, 112, 405, 279, 376, 341, 306, 348, 368, 244, 370, 293, 259,
       402, 496, 282, 367, 382, 230, 389, 245, 285, 265, 227, 105, 304,
       126, 299, 314, 401, 400, 398, 319, 284,  23, 307, 313, 290, 399,
       118,  10, 310]

index2 = ind2[:360]
w_gen, b_gen = calAccuracy(X_train[:,index2], Y_train)

saveCon(X_test[:,index2], w_gen, b_gen)