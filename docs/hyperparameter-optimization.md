# Hyperparameter Optimization Using BOHB (Bayesian Optimization and Hyperband)

# Team Members: Peter Akdemir, Joshua Quizon

## Hyperparameter Optimization
When training a model, certain values will be tweaked by the model itself.  For example, one of these values could be the weights associated with any sort of regression model.  In the logistic regression homework assignment, the main idea was to train the model (sigmoid) by updating the weights associated with each feature through gradient descent.  These trainable values are called model parameters.<br>

It is important to note that there is another category of variables in supervised machine learning that cannot be trained and are rather set to specific values that stay constant during a training session.  These values are parameters, and more specifically are hyperparameters.  Below is a list of hyperparameters that have been mentioned in lectures and assignments:
1. Learning rate
2. Number of epochs
3. Batch size
4. Dropout rate
5. Depth of a decision tree
6. Number of neurons in a neural network
7. Number of convolutional layers
8. Number of kernels in a filter
9. Number of filters in a layer 

Depending on the overall model, a wide variety of hyperparameters will be in play.  Specifically for this project, the clear hyperparameters that are not completely hidden inside the UNet black box are the learning rate **(idk about this tbh cuz i can't find a learning rate anywhere in the model)**, the number of epochs, batch size, and dropout rate. <br>

From running the baseline test on the model, the main issue that was noticed was the model was overfitting the training set.  As stated in the markdown file [baseline-performance.md](baseline-performance.md), this could be due to the number of epochs being set to too high of a number (100 in the baseline test).  Other properties were unexplored at the time when baseline-performance.md was written (during milestone-2), but will be explored in this milestone.  These other properties of interest are dropout rate and batch size.  The baseline values for these hyperparameters are as follows:
- dropout rate: 0.2 (applies to all hidden layers)
- batch size: 16

A lower dropout rate and batch size can provide implicit regularization to the model, as the model will be given smaller input and each layer will output less--this will increase the noise of the overall output and may ultimately lead to a more general model.  However, the values for dropout rate and batch size are already quite low.  Therefore, hyperparameter optimization will be utilized to find the most suitable values for the three hyperparameters of interest in this project.  The specific hyperparameter optimization strategy for Team 7 is BOHB (Bayesian Optimization and Hyperband) which combines the accuracy of Bayesian Optimization and the speed of Hyperband.  Both of the components that make up BOHB will be explained at a high level below.
## Bayesian Optimization
TODO

## Hyperband
TODO

## Combining the 2 Hyperparameter Optimization Algorithms
TODO

## Resulting BOHB Optimized Hyperparameters
TODO

## Training and Validation Epochs vs Loss
TODO

## 10 Prediction Results
TODO

## Precision and Recall Values
TODO
