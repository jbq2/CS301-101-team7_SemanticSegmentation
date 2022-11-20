# Hyperparameter Optimization Using BOHB (Bayesian Optimization and Hyperband)

# Team Members: Peter Akdemir, Joshua Quizon
# HPO Method for Team 7: BOHB (Method 7)

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

Depending on the overall model, a wide variety of hyperparameters will be in play.  Specifically for this project, the clear hyperparameters that are not completely hidden inside the UNet black box are the learning rate, the number of epochs, type of optimizer, and batch size. <br>

From running the baseline test on the model, the main issue that was noticed was the model was overfitting the training set.  As stated in the markdown file [baseline-performance.md](baseline-performance.md), this could be due to the number of epochs being set to too high of a number (100 in the baseline test).  Other properties were unexplored at the time when baseline-performance.md was written (during milestone-2), but will be explored in this milestone.  These properties of interest are learning rate, dropout rate, batch size, and optimizer.  The baseline values for these hyperparameters are as follows:
- learning rate: 0.001 (adam optimizer)
- batch size: 16
- optimizer: adam
- SGD momentum: 0.01 (by defaut if using SGD as an optimizer)

It is important to note that the optimizer can either be adam or SGD--both of these use a learning rate.  However, if SGD is chosen, the SGD momentum hyperparameter must also be considered.<br>

The learning rate is set to a rather lower number--lower learning rates tend to increase the risk of overfitting.  The lower batch size can provide implicit regularization to the model, as the model will be given smaller input--this will increase the noise of the overall output and may ultimately lead to a more general model.  However, the value for batch size is already quite low.  During the baseline performance, the optimizer that was used was adam.  In this milestone, SGD will also be considered as a possible optimizer, which in turn brings the SGD momentum hyperparameter into play.  Therefore, hyperparameter optimization will be utilized to find the most suitable values for the four hyperparameters of interest in this project.  The specific hyperparameter optimization strategy for Team 7 is BOHB (Bayesian Optimization and Hyperband) which combines the accuracy of Bayesian Optimization's informed search and the speed of Hyperband.  Both of the components that make up BOHB will be explained at a high level below.

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

## Resulting BOHB Optimized Hyperparameters Using HpBandster's BOHB Package
### Test Results after 15 successive halving operations with BOHB
![image](https://user-images.githubusercontent.com/98120760/202926943-29f08607-af38-429f-8e84-d2c21ef6169a.png)
- A short test performed by Joshua Quizon
- This was the result of running BOHB for n = 5 iterations, which resulted in 15 successive halving iterations
- The resulting data depicts the hyperparameter configuration space that is used for each line--this configuration was used to train and assess the model for that successive halving iteration
- Note: SGD momentum is only used when the optimizer was SGD; if SGD is not used, the SGD momentum is ignored

### Results after 35 successive halving operations with BOHB
![image](https://user-images.githubusercontent.com/98120760/202926892-befbd9bd-3746-48e8-bff5-8bc4c43e1c59.png)
- A BOHB run performed by Peter Akdemir
- This was the result of running BOHB for 35 successive halving iterations
- The resulting data, again, depicts each hyperparameter configuration and the model's loss from using that configuration
- The configuration on the last line contains the most optimal hyperparameter configuration
    - Learning rate: 0.004740049723020277
    - Batch size: 9
    - Optimizer: Adam

## Training and Validation Epochs vs Loss
![image](https://user-images.githubusercontent.com/98120760/202927013-d0391ec1-997a-4c58-9634-b0b637c027d9.png)
- Comparing this training and validation epochs vs loss graph from the one produced in milestone 2, it can be seen that the model is not overfitting the training data as much anymore
- This is likely due to the ability to find the most optimal hyperparameters
- Although the loss is not nearly as low as the one from milestone 2, there is only a small difference
- Despite the slightly higher loss coming from the "optimal" hyperparameters, the model is no longer overfitting the training set which implies a more generalized model 

## Precision and Recall Values
![image](https://user-images.githubusercontent.com/98120760/202927153-9c682fcc-b982-4b94-94e7-473bbc716b5a.png)
- Much like the precision and recall graph from milestone 2, the above graph is quite chaotic
- There is not much area under the curve which may imply that the model is lacking performance, but this is rather contradictory:
    - Recall values increase as the number of epochs increase--this is the desired and expected behavior
    - Precision values fluctuate, but ultimately converge to values of around 0.81 to 0.84
- The reason for the abnormal precision and recall curve is due to the fluctuations of the precision and recall values at earlier epochs when training the model

## 10 Best Prediction Results
![image](https://user-images.githubusercontent.com/98120760/202927065-fa1830d4-7d5e-4f0c-ac8a-c40b4e5fd582.png)
![image](https://user-images.githubusercontent.com/98120760/202927084-a8d3a7e8-3429-4b3e-9589-74664537addc.png)
![image](https://user-images.githubusercontent.com/98120760/202927105-3d713d2f-6187-47e6-9b96-a700bd64a5b1.png)
![image](https://user-images.githubusercontent.com/98120760/202927113-c3f95f69-a980-4e88-8925-a4ce837890bc.png)
