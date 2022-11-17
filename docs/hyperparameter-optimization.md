# Hyperparameter Optimization Using BOHB (Bayesian Optimization and Hyperband)

# Team Members: Peter Akdemir, Joshua Quizon

## Hyperparameter Optimization
When training a model, certain values will be tweaked by the model itself.  For example, one of these values could be the weights associated with any sort of regression model.  In the logistic regression homework assignment, the main idea was to train the model (sigmoid) by updating the weights associated with each feature through gradient descent.  These trainable values are called model parameters.<br>

It is important to note that there is another category of variables in supervised machine learning that cannot be trained and are rather set to specific values that stay constant during a training session.  These values are parameters, and more specifically are hyperparameters.  Below is a list of hyperparameters that have been mentioned in lectures and assignments:
1. Learning rate
2. Number of epochs
3. Batch size
4. Optimizer
5. Dropout rate
6. Depth of a decision tree
7. Number of neurons in a neural network
8. Number of convolutional layers
9. Number of kernels in a filter
10. Number of filters in a layer 

Depending on the overall model, a wide variety of hyperparameters will be in play.  Specifically for this project, the clear hyperparameters that are not completely hidden inside the UNet black box are the learning rate, the number of epochs, batch size, and dropout rate. <br>

From running the baseline test on the model, the main issue that was noticed was the model was overfitting the training set.  As stated in the markdown file [baseline-performance.md](baseline-performance.md), this could be due to the number of epochs being set to too high of a number (100 in the baseline test).  Other properties were unexplored at the time when baseline-performance.md was written (during milestone-2), but will be explored in this milestone.  These other properties of interest are learning rate, dropout rate, batch size, and optimizer.  The baseline values for these hyperparameters are as follows:
- learning rate: 0.001 (adam optimizer)
- dropout rate: 0.2 (applies to all hidden layers)
- batch size: 16
- optimizer: adam
- SGD momentum: 0.01 (by defaut if using SGD as an optimizer)

It is important to note that optimizer can either be adam or SGD--both of these use a learning rate.  However, if SGD is chosen, the SGD momentum hyperparameter must also be considered.<br>

The learning rate is set to a rather lower number--lower learning rates tend to increase risk of overfitting.  A lower dropout rate and batch size can provide implicit regularization to the model, as the model will be given smaller input and each layer will output less--this will increase the noise of the overall output and may ultimately lead to a more general model.  However, the values for dropout rate and batch size are already quite low.  Therefore, hyperparameter optimization will be utilized to find the most suitable values for the four hyperparameters of interest in this project.  During the baseline performance, the optimizer that was used was adam.  In this milestone, SGD will also be considered as a possible optimizer, which in turn brings the SGD momentum hyperparameter into play.  The specific hyperparameter optimization strategy for Team 7 is BOHB (Bayesian Optimization and Hyperband) which combines the accuracy of Bayesian Optimization's informed search and the speed of Hyperband.  Both of the components that make up BOHB will be explained at a high level below.

## Bayesian Optimization and Hyperband (BOHB)
Solid hyperparameters are important because they can make or break a model.  Too high of a learning rate will likely cause too much noise for the model, thus causing underfitting.  If too many epochs are used, the model will likely overfit the training data, thus leading to a high difference in training and validation loss.  While a programmer can brute force the finding of the right hyperparameters, such a task is tedious and may result in never actually finding the most optimal hyperparameter configurations.  BOHB is a technique that uses aspects of two different hyperparameter optimization techniques: Bayesian Optimization and Hyperband.  Essentially, BOHB makes use of Bayesian Optimization's awareness of previous calculations and Hyperband's speed to optimize hyperparameters in an efficient manner.  Before getting into the BOHB algorithm, it is important to briefly go over Bayesian Optimization and Hyperband first.

### Bayesian Optimization (BO)
Consider hyperparameter configuration space $\mathbb{H}$; this is essentially, the proposed "best" values for each hyperparameter in question.  These individual values can be inputted in some black box function $f$.  The goal of BO is to approximate $f$, which will in turn lead to the best possible hyperparameters for the model.  The output of $f(\mathbb{H})$ is the error on the validation set--minimizing this error implies that the approximate function is close to the true black box function.<br>

There are two functions that are important for BO:
- Surrogate function: this is the function that will constantly be updated, and will eventually act as the approximate to the black box function
- Acquisition function: looks for the next set of hyperparameters to test that can potentially lead to a lower error on the validation set (the issue here is that this is sort of a blind guess because the shape of $f$ is not known, as it is a black box)

An example of an acquisition function is Expected Improvement, which describes areas in which there the surrogate function is return high error, and those are areas of improvement.  Therefore, the strategy with Expected Improvement is to take find the value $x$ that leads to the maximum value $a(x)$ that Expected Improvement will return--this will be the next hyperparameter setting to check for.

Bayesian Optimization tends to converge in less iterations compared to other optimization methods.  This is because BO is an informed search method--the next set of hyperparameter configurations to be tested is based on the previously tested configurations.  However, Bayesian Optimization still takes a while because each evaluation for the next configuration is expensive--you must calculate the next best value to test for each hyperparameter and also update the surrogate function.

### Hyperband
Consider multiple hyperparameters, each having a set of possible values that will be tested.  If 3 hyperparameters have 10 possible values each, then 1000 possible configurations need to be considered.  This number increases greatly as the number of hyperparameters and possible values for each hyperparameter increase.  Therefore, hyperparameter optimization can be a slow and tedious process.  Hyperband aims to lessen the time it takes to evaluate possible hyperparameter configurations.  The basic steps for Hyperband are as follows:
- Sample a set of configurations from the set of all configurations
- The model is trained with each set of configurations
    - scrap the configurations that lead to relatively bad performance
    - keep the configurations that lead to relatively good performance, focusing on additional hyperparameters in the configurations

The above steps prompt the following question: what are the "good" and "bad" configurations?  Another aspect of Hyperband is that it uses Successive Halving.  In this procedure, 50% of the sampled configurations are scrapped, while 50% are moved on to the next phase.  At each iteration of Hyperband, the sample size of configurations to be tested is cut in half until there is only 1 configuration left.  The issue with Successive Halving is that it is a hyperparameter optimizer that also has hyperparameters.  For Successive Halving, you must decide the number of input configurations, how long you would like to train the model based on those configurations (budget), and how many times you want to cut the sample in half.<br>

Hyperband fixes the above issue of Successive Halving's lack of robustness by trying multiple splitting settings.  The two extremes of splitting are (1) not splitting the samples at all and (2) splitting the samples until there is only 1 configuration left.  Hyperband tries both of those extremes, as well as the splitting settings in between the extremes.  In other words, Hyperband tries multiple ways of splitting the samples and assesses the output configurations that survived the splits. However, Hyperband is still susceptible to the fact that the input configurations are sampled at random.  Therefore, convergence to an "optimal" configuration may not lead to the most optimal hyperparameter values.
### BOHB Algorithm
BOHB combines the informed search of Bayesian Optimization with the speed of Hyperband.  The process is as follows:
- Use Hyperband; but store the validation information regarding each (configuration, budget) pair
- Once enough pairs and their validation scores have been collected, fit a surrogate model to it
- With the existence of a surrogate model, use an acquisition function to find the specific hyperparameter values that yield the maximum of the acquisition function
- Continue to random sample using Hyperband, as this updates the surrogate model by adding some noise to it which will increase the overall model's generalization and prevent it from overfitting

## Resulting BOHB Optimized Hyperparameters
TODO

## Training and Validation Epochs vs Loss
TODO

## 10 Prediction Results
TODO

## Precision and Recall Values
TODO
