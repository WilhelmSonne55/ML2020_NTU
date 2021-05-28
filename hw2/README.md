# Predict the income of a man exceeds $50000 or not
- data: could get the training data at\
https://www.kaggle.com/c/ml2020spring-hw2/overview

-hw2_generative.py: generative training model for classification\
-hw2_logistic.py: discriminative training model for classification\
-hw2_logistic.sh: script for hw2_logistic.py\
-hw2_generative.sh: script for hw2_generative.py\
e.g. \
hw2_logistic.sh ./data/train.csv ./data/test_no_lable.csv ./data/X_train ./data/Y_train ./data/X_test ./out.csv\
hw2_generative.sh ./data/train.csv ./data/test_no_lable.csv ./data/X_train ./data/Y_train ./data/X_test ./out.csv

# Generative and Discriminative 
discriminative model's accuracy is better
generative model is limited by the distribution of training data.
If the distribution of training data doesn't match the actual distribution,
the result will not be very great.

- discriminative model:\
![image](https://user-images.githubusercontent.com/13451511/119992023-acff2c00-bffc-11eb-8a91-583284b677cc.png)
- generative model:\
![image](https://user-images.githubusercontent.com/13451511/119992081-c011fc00-bffc-11eb-84b4-ea87bb07787b.png)

# Iteration, Accuration & Loss
The accuracy will increase by the loss which is descrease by the iteration number.

![image](https://user-images.githubusercontent.com/13451511/119993349-0caa0700-bffe-11eb-9a0c-56d2339e3a56.png)

# Regularization and Lambda
It is obvious overfitting without regularization term.
Lambda is better at 0.02 after then it becomes worse.\
![image](https://user-images.githubusercontent.com/13451511/119993673-69a5bd00-bffe-11eb-984d-db662eaa7206.png)

After select feature, the lambda is much smaller.
The reason might be the model nearly converges.
![image](https://user-images.githubusercontent.com/13451511/119993995-c0ab9200-bffe-11eb-80a2-5b24e55cee47.png)

# Feature Select
I choose the features based on the weight.\
And select top 0-170 features to be the training data.
