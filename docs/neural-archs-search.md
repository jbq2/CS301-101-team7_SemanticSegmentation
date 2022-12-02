# Neural Architecture Search
# Team Members: Peter Akdemir, Joshua Quizon

# Knowledge Distillation
## Current State of the Model
<p>
Machine learning models are extremely powerful programs that can recognize structure
within data sets.  An example is this current project, where a multi-class UNET model
is used to segment images and classify different landmarks/structures/land types
in images.  So far, what has been done was DigitalShreeni's UNET model and data preprocessing
methods were used to set up the data and record baseline performances of the model.  Then, 
hyperparameter optimization was performed to find the optimal hyperparameter values that would
yield a lower loss and less overfitting.
</p>

<p>
However, it is important to note that for previous 2 milestones (milestone 2 and 3), the tasks
required the use of Colab's premium GPUs.  If they were not used, testing the program and training
the model for milestone 2 would have taken weeks.  Without the premium GPUs, each epoch for training
would have taken 15-20 minutes to complete.  Even with the GPUs for milestone 3, BOHB hyperparameter
optimization took over 1 hour to complete only 20 iterations.  The issue with the current model is 
that it is massive and its operations are extremely excessive.  Clearly, the current state of the 
model is unfit for production purposes.
</p>

<p>
The goal of milestone 4 is to relieve this issue.  In a production setting, it is not guaranteed that
the user or client will have state of the art GPUs for training the model or making predictions with it.
Therefore, the **knowledge distillation** will be used to minimize the complexity/excessiveness of the
model's operations.
</p>

## Explanation of Knowledge Distillation
<p>
In knowledge distillation, a smaller model is built with the same functionalities as the larger model, except
it is more "compact".  With the case of the UNET model, a possible smaller model could be implemented with 
a lower number of filters for each convolutional layer and upscaling layer.  However, because the smaller model
still functions under the same interface as the larger UNET model, its classification of the data should remain
the same.  However, due to the smaller size of it, it must be trained with different data that suits it.
</p>

<p>
To preface the explanation of knowledge and how it can be applied to this scenario, recall that UNET
is a model comprised on an ensemble of smaller, simpler models.  It is a neural network, therefore it 
contains numerous neurons.  Therefore, according to the seminal paper, the smaller model can be trained 
with the probabilities of each class produced/predicted by the larger model.  These probabilities are termed as
"soft targets" for the smaller model, and will be used to train it.  There is a chance for the soft targets to yield
high entropy which implies greater information gain for each data point.  This allows the smaller model to be trained
with smaller data sets.
</p>  

<p>
However, it is stated in the seminal paper that this may be an issue because specific data points may contain very low 
probabilities for certain classes.  To resolve this problem, the logit values of the probabilities will
be used rather than the raw probabilities themselves.  According to the paper, the class probabilities produced
by a neural network put the logit values through a softmax layer which then outputs the probability of each class:
$$q_i = \frac{exp(z_i/T)}{sum_{j}^{}exp(z_j)/T}$$
</p>