import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import csv

def calCost(x,w,y):
  col_size = x.shape[0]
  test_x = np.concatenate((np.ones([col_size, 1]), x), axis = 1).astype(float)
  y_ = np.dot(test_x, w)
  error = np.sum((y - y_)**2)/(2*col_size)
  return error
  
  
def pltloss(y_loss, x_iter, learning_rate):
  plt.plot(x_iter, y_loss, label =  'learning_rate:' + str(learning_rate))
  plt.ylabel('Loss')
  plt.xlabel('Iteration')
  
  
# Adagrad
def calWeightAdagrad(x, y, x_val, y_val, iter_time, learning_rate, plot = False):
  dim_col, dim_row = x.shape
  w = np.zeros([dim_row+1, 1])
  x_train = np.concatenate((np.ones([dim_col, 1]), x), axis = 1).astype(float)

  adagrad = np.zeros([dim_row+1, 1])
  eps = 0.0000000001
  min_error = 1000
  early_stop_cnt = 0
  if plot == True:
    y_loss = list()
    x_iter = list() 
  for t in range(iter_time):
      
      loss = np.sum(np.power(np.dot(x_train, w) - y, 2))/dim_col#rmse
      if t%100==0 and plot == True:
        #print('loss:',loss)
        x_iter.append(t)
        y_loss.append(loss)

      gradient = 2 * np.dot(x_train.transpose(), np.dot(x_train, w) - y)  #dim*1
      adagrad += gradient ** 2
      w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
      cost_val = calCost(x_val, w, y_val)
      if(cost_val < min_error):
        min_error = cost_val
        early_stop_cnt = 0
      else:
        early_stop_cnt +=1

      if early_stop_cnt > 200:
        print('early stop!!')
        break
  print('Finished training after {} iter:', t)
  if plot == True:
    pltloss(y_loss, x_iter, learning_rate)
  return w, cost_val
  
def select_model(x,y,index, pow):
  x_choose = np.empty([x.shape[0], 9*len(index)])
  g = 0

  # xi
  for i in index:
    x_choose[:,g*9:g*9+9] = x[:, i*9:i*9+9]
    g +=1

  # xi^2
  for i in index:
    x_choose = np.append(x_choose, np.power(x[:, i*9:i*9+9], pow), axis = 1)

  #pm2.5 x9*xi
  for i in index:
    k = np.multiply(x[:, 9*9:9*9+9], x[:, i*9:i*9+9])
    x_choose = np.append(x_choose, k, axis = 1)
  y_choose = y
  return x_choose, y_choose

def saveSubmit(ans_y):
  with open('submit.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)

# pre-processing
data = pd.read_csv('./data/train.csv', encoding = 'big5')  
data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()

month_data = {}
for month in range(12):
    sample = np.empty([18, 480]) # 18 datas each day
    for day in range(20): # only 20 days per months
        # 24 hours
        sample[:, day * 24 : (day + 1) * 24] = raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]
    month_data[month] = sample

x = np.empty([12 * 471, 18 * 9], dtype = float)
y = np.empty([12 * 471, 1], dtype = float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14: #20days:0-19, 24hours:0-23, only support 471 index = 19*24 + 24-9
                continue
            x[month * 471 + day * 24 + hour, :] = month_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1, -1) #vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9] #value 9th is PM2.5 in features
	
mean_x = np.mean(x, axis = 0) #18 * 9 
std_x = np.std(x, axis = 0) #18 * 9 
for i in range(len(x)): #12 * 471
    for j in range(len(x[0])): #18 * 9 
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]

# feature scaling
x_train_set = x[: math.floor(len(x) * 0.8), :]
y_train_set = y[: math.floor(len(y) * 0.8), :]
x_validation = x[math.floor(len(x) * 0.8): , :]
y_validation = y[math.floor(len(y) * 0.8): , :]
 
# pre-processing test data
testdata = pd.read_csv('./data/test.csv', header = None, encoding = 'big5')
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy() #4320x9
test_x = np.empty([240, 18*9], dtype = float) #=> 4320/18 = 240
for i in range(240):
    test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)  # i col <= 18x9
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)

# train
x_train_n, y_train_n = select_model(x_train_set, y_train_set, range(0,18), 2)
x_val_n, y_val_n = select_model(x_validation, y_validation, range(0,18), 2)
#w_p, error = calWeightGD(x_train_n, y_train_n, x_val_n, y_val_n, 120000, 0.0016, True)
w_p, error = calWeightAdagrad(x_train_n, y_train_n, x_val_n, y_val_n, 20000, 2, True)
np.savez('model.npz', w = w_p, std = std_x, mean = mean_x)

# predict
x_test_n, y_test_n = select_model(test_x, 0, range(0,18), 2)
x_test_n = np.concatenate((np.ones([x_test_n.shape[0], 1]), x_test_n), axis = 1).astype(float)
test_y_sel = np.dot(x_test_n, w_p)
saveSubmit(test_y_sel)