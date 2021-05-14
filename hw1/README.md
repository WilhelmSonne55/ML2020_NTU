

# Predict 10th hour PM2.5
18 feature\
train.csv:data in 20 days each month\
test.csv: data in 9 hours 


## The relation between learning rate and iteration times:
  - If learning rate is too big, the value of loss might go up.\
![image](https://user-images.githubusercontent.com/13451511/118279419-b2894c00-b4fd-11eb-974d-eae22b2317e4.png)

## The difference between 5 hours(5-9) and 9 hours(1-9) data:
 - The 5 hours data will be easier to converge. 
   The latter part of data might be similar.
 - However, the loss of 9 hours data is lower.
   I think more features could make it easier to ignore the noise.
   
![image](https://user-images.githubusercontent.com/13451511/118282919-60e2c080-b501-11eb-8f23-16a0c7481fb6.png)

![image](https://user-images.githubusercontent.com/13451511/118282881-56c0c200-b501-11eb-8149-3873baa672da.png)

## The difference between only 9th feature and 18 features
 - More features are easier to get better prediction.\
   Moreover, according to the iteration times, the noise could be ignored.
![image](https://user-images.githubusercontent.com/13451511/118283770-4a893480-b502-11eb-9e1c-3c522416aab3.png)

## The difference between Adagrad and Normal Gradient Descent
 - Gradient Descent
   1.learning rate should be very small or the result will not converge.\
    The difference is devide power of gradient which could prevent the value too big.
 - Adagrad
   The iteration time is much smaller.\
   The big learning rate could be the reason.
 Although, these hpyer-parameters are different. \
 Both results are close.
 ![image](https://user-images.githubusercontent.com/13451511/118284861-7a850780-b503-11eb-9d89-e49658b56932.png)
