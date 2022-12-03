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
contains numerous neurons.  Therefore, the smaller can be trained on a transfer set (which can be the
original training set) while also considering the probability distributions of the classes for each
case in the transfer set.  These probabilities are termed as "soft targets" for the smaller model.  There is a chance for the soft targets to yield
high entropy which implies greater information gain for each data point.  This allows the smaller model to be trained
with smaller data sets.
</p>  

<p>
However, it is stated in the seminal paper that this may be an issue because specific data points may contain very low 
probabilities for certain classes.  To resolve this problem, the logit values of the probabilities will
be used rather than the raw probabilities themselves.  According to the paper, the class probabilities produced
by a neural network put the logit values through a softmax layer which then outputs the probability of each class:
$$q_i = \frac{exp(z_i/T)}{\sum_{j}^{}exp(z_j)/T}$$
</p>

<p>
A main improvement to this method is to train the smaller, "distilled" model to produce the correct labels for the data.
A way to do this is by using the correct labels to modify the soft targets--this will allow the distilled model to 
produce probability distributions that indicate the likelihood of a case to be a certain class.  However, the authors of 
the seminal paper propose a different method using 2 objective functions: 1st is a cross entropy function with the soft targets,
and 2nd is the cross entropy with the correct labels.  As will be explained in the next section, the Colab Notebook associated with this project attempts to implement the method of knowledge distillation discussed in the seminal paper. 
</p>

## Implemention Of Knowledge Distillation
<p>
The first step in the implementation involves training the original model which will act as a teacher model.  The following is a short summary of the information regarding the teacher model:

- teacher model is a UNET model returned by the function multi_teacher_unet_model
- teacher model has 10 layers
    - 4 downscaling (c1, c2, c3, c4)
        - c1: 16 filters
        - c2: 32 filters
        - c3: 64 filters
        - c4: 128 filters
    - a final downscaling layer with 256 filters, no max pooling (c5)
    - 4 upscaling layers (u6, u7, u8, u9)
        - u6: 128 filters
        - u7: 64 filters
        - u8: 32 filters
        - u9: 16 filters
    - 1 output layer (activation function being softmax)
- the teacher outputs softmax values

The teacher needs to be trained because it will provide predictions to batched datasets.  These predictions will act as the knowledge that will be "distilled" so that the student can train with it.
</p>

<p>
The second step is to implement a student model.  Generally, student models will have less trainable parameters than the teacher model to it being "smaller" and more compact.  For this project, the student model has less filters per layer.  Below is a short summary of the information regarding the student model:

- student model is also a UNET model, returned by the function multi_student_unet_model
- model has the same number of layers (10):
    - 4 downscaling layers with maxpooling (c1, c2, c3, c4)
        - c1: 4 filters
        - c2: 8 filters
        - c3: 16 filters
        - c4: 32 filters
    - a final downscaling layer with 64 filters, no max pooling (c5)
    - 4 upscaling layers (u6, u7, u8, u9)
        - u6: 32 filters
        - u7: 16 filters
        - u8: 8 filters
        - u9: 4 filters
    - 1 output layer (activation function being softmax)
- the student also outputs softmax values

</p>

<p>
The next step is to build the Distiller class which extends keras's Model class.  This Distiller class essentially trains the student model given the teacher's predictions.  Since Distiller is a subclass of Model, it inherits the functions defined in Model, and also overrides some of those methods (as it should because Distiller must train the student model differentyl which requires a different implementation).  The following is a list of overridden methods and a brief overview of what they do:

- compile: does what compile would normally do with a regular keras Model object.  It accepts new parameters that were not experimented with before this milestone; student_loss_fn, distillation_loss_fn, alpha, and temperature
- call: a function that implements the forward pass of the Model; it is implicitly called by doing model(x_data, training=True/False)
- train_step: implicitly called by fit().  train_step is called whenever the model must be trained with a new batch of data.
- test_step: implicitly called by evaluate(), or fit() when the validation_data parameter is set

The most important function that was implemented is train_step, as this is the inner workings of actuall training the model and fitting data to it.  The process of Colab notebook's train_step implemention in Distiller is as follows:
1. pass data to it
2. predict using the teacher model
3. predict using the student model
4. find the distillation loss using the distillation_loss_fn, and multiply the result by the squared temperature $T^2$
5. find the overall loss with the following equation: $loss = \alpha \cdot loss_{student} + (1 - \alpha) \cdot loss_{distillation}$
6. compute the gradient with respect to each of the student's trainable variables
7. apply the gradients to the used optimizer (Adam in this case) 
8. return the results of that single pass (student_loss and distillation_loss)
</p>