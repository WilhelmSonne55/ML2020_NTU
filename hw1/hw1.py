import sys
import pandas as pd
import numpy as np
import csv

# model select
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

#save csv
def saveSubmit(ans_y, output):
  with open(output, mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)

# load csv files
testfile = sys.argv[1]
outputfile = sys.argv[2]
		
# pre-processing test data
testdata = pd.read_csv(testfile, header = None, encoding = 'big5')
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy() #4320x9
test_x = np.empty([240, 18*9], dtype = float) #=> 4320/18 = 240
for i in range(240):
    test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)  # i col <= 18x9
	
# load weight, std, mean
arch = np.load('model.npz')
w_p = arch['w']
std_x = arch['std']
mean_x = arch['mean']

for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)	

# predict
x_test_n, y_test_n = select_model(test_x, 0, range(0,18), 2)
x_test_n = np.concatenate((np.ones([x_test_n.shape[0], 1]), x_test_n), axis = 1).astype(float)
test_y_sel = np.dot(x_test_n, w_p)
saveSubmit(test_y_sel, outputfile)