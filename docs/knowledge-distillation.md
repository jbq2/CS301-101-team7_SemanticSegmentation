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
However, it is important to note that for the previous 2 milestones (milestones 2 and 3), the tasks
required the use of Colab's premium GPUs.  If they were not used, testing the program and training
the model for milestone 2 would have taken weeks.  Without the premium GPUs, each epoch for training
would have taken 15-20 minutes to complete.  Even with the GPUs for milestone 3, BOHB hyperparameter
optimization took over 1 hour to complete only 20 iterations.  The issue with the current model is 
that it is massive and its operations are extremely excessive.  The current state of the 
model is unfit for production purposes.
</p>

<p>
The goal of milestone 4 is to relieve this issue.  In a production setting, it is not guaranteed that
the user or client will have state-of-the-art GPUs for training the model or making predictions with it.
Therefore, the **knowledge distillation** will be used to minimize the complexity/excessiveness of the
model's operations.
</p>

## Explanation of Knowledge Distillation
<p>
In knowledge distillation, a smaller model is built with the same functionalities as the larger model, except
it is more "compact".  In the case of the UNET model, a possible smaller model could be implemented with 
a lower number of filters for each convolutional layer and upscaling layer.  However, because the smaller model
still functions under the same interface as the larger UNET model, its classification of the data should remain
the same.  However, due to its smaller size of it, it must be trained with different data that suit it.
</p>

<p>
To preface the explanation of knowledge and how it can be applied to this scenario, recall that UNET
is a model comprised of an ensemble of smaller, simpler models.  It is a neural network, therefore it 
contains numerous neurons.  Therefore, the smaller can be trained on a transfer set (which can be the
original training set) while also considering the probability distributions of the classes for each
case in the transfer set.  These probabilities are termed "soft targets" for the smaller model.  There is a chance for the soft targets to yield
high entropy which implies greater information gain for each data point.  This allows the smaller model to be trained
with smaller data sets.
</p>  

<p>
However, it is stated in the seminal paper that this may be an issue because specific data points may contain very low 
probabilities for certain classes.  To resolve this problem, the logit values of the probabilities will
be used rather than the raw probabilities themselves.  According to the paper, the class probabilities produced
by a neural network put the logit values through a softmax layer which then outputs the probability of each class:
$$q_i = \frac{exp(z_i/T)}{\sum_{j}^{}exp(z_j)/T}$$
In the teacher model (which is the same model used in the previous milestones), the softmax probabilities are returned because the output layer's activation function is set to 'softmax.'
</p>

<p>
The main improvement to this method is to train the smaller, "distilled" model to produce the correct labels for the data.
A way to do this is by using the correct labels to modify the soft targets--this will allow the distilled model to 
produce probability distributions that indicate the likelihood of a case being a certain class.  However, the authors of 
the seminal paper propose a different method using 2 objective functions: 1st is a cross entropy function with the soft targets,
and 2nd is the cross entropy with the correct labels.  As will be explained in the next section, the Colab Notebook associated with this project attempts to implement the method of knowledge distillation discussed in the seminal paper. 
</p>

## Implemention Of Knowledge Distillation
<p>
The first step in the implementation involves training the original model which will act as a teacher model.  The following is a summary of the information regarding the teacher model:

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
The second step is to implement a student model.  Generally, student models will have fewer trainable parameters than the teacher model to it being "smaller" and more compact.  For this project, the student model has fewer filters per layer.  Below is a summary of the information regarding the student model:

- student model is also a UNET model, returned by the function multi_student_unet_model
- the model has the same number of layers (10):
    - 4 downscaling layers with max-pooling (c1, c2, c3, c4)
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
The next step is to build the Distiller class which extends keras's Model class.  This Distiller class essentially trains the student model given the teacher's predictions.  Since Distiller is a subclass of Model, it inherits the functions defined in Model, and also overrides some of those methods (as it should because Distiller must train the student model differently which requires a different implementation).  The following is a list of overridden methods and a brief overview of what they do:

- compile: does what compile would normally do with a regular keras Model object.  It accepts new parameters that were not experimented with before this milestone; student_loss_fn, distillation_loss_fn, alpha, and temperature
- call: a function that implements the forward pass of the Model; it is implicitly called by doing model(x_data, training=True/False)
- train_step: implicitly called by fit().  train_step is called whenever the model must be trained with a new batch of data.
- test_step: implicitly called by evaluate(), or fit() when the validation_data parameter is set

The most important function that was implemented is train_step, as this is the inner workings of training the model and fitting data to it.  The process of the Colab notebook's train_step implementation in Distiller is as follows:
1. pass data to it
2. predict using the teacher model, outputs softmax probabilities; divide softmax matrix by the temperature
3. predict using the student model, outputs softmax probabilities; divide softmax matrix by the temperature
4. find the distillation loss using the distillation_loss_fn, and multiply the result by the squared temperature $T^2$
5. find the overall loss with the following equation: $loss = \alpha \cdot loss_{student} + (1 - \alpha) \cdot loss_{distillation}$
6. compute the gradient with respect to each of the student's trainable variables
7. apply the gradients to the used optimizer (Adam in this case) 
8. return the results of that single pass (student_loss and distillation_loss)
</p>


# Student Results

## Important Notes Regarding Training Issues:
- there was quite some difficulty getting the teacher and student to train properly
- on some instances, the models seemed to not learn at all, while in other instances they did seem to train
- a pattern was found that if the number of epochs was set too high, the teacher would not learn properly (the accuracy of the model would stay at 0.5308, and the loss would also stay around a specific value no matter how many epochs the model was trained for)
- as a result, teacher training was limited to 5 epochs
- the only way for the student to be trained properly is if it is given proper predictions from a well-trained/functioning teacher model
- due to limited compute units left, the student could only train for 15 epochs from a teacher that was trained for 5 epochs (due to issues mentioned in the previous bullets)

## Training and Validation Loss Plot:
![image](https://user-images.githubusercontent.com/98120760/205477387-e135ec6e-b752-41a1-a6ca-fc87a0175c29.png)
- the training and validation loss plot is quite strange for this milestone
- it is likely that the plot is not a great indicator of the relationship between the two values over the number of epochs due to the issues mentioned above
- the plot depicts the training and validation loss for 15 epochs
- the training loss is far greater than the validation loss which is a very unlikely outcome
- it might be because a different loss function (sparse categorical cross entropy) was used instead of total_loss
- while the training and validation loss plot and values fail to demonstrate the performance of the student model, the predictions in the next subsection show that the student was indeed trained


## Student Predictions
![image](https://user-images.githubusercontent.com/98120760/205477411-a54d8af5-6857-4f74-8f89-b0d1bb535429.png)
![image](https://user-images.githubusercontent.com/98120760/205477436-695b6bbe-c98f-4d05-b27b-8fa456f95b99.png)
![image](https://user-images.githubusercontent.com/98120760/205477448-e1d92b51-79d0-4cd2-86f9-2d326826e294.png)
![image](https://user-images.githubusercontent.com/98120760/205477459-778dd5e9-b36d-4336-a9d5-a7a87a51b9a4.png)
- above are the predictions of the student model
- the predictions are not the best; regardless, the student was able to pick up most details and highlight the general areas of interest (roads, vegetation, water, etc.)
- if the model had been trained for longer, the predictions would have likely been much better
- however, as stated above, there were issues with training the teacher and student model for too long and there was also the lack of Colab compute units to experiment more and discover the underlying bug in the implementation

## Precision and Recall:
- these values were not readily available due to the usage of sparse categorical cross entropy as the loss function for the teacher and student models
- a switch this loss function was required to calculate the distillation loss of the Distiller properly
- sparse categorical cross entropy does not evaluate on one hot encoded ground truths
- precision and recall values are built from one hot encoded ground truths
- unfortunately, this requirement could not be met in this milestone
- for the state of this milestone, it is suspected that the precision and recall values would be rather low (due to the small amount of training the student had)
- however, if trained for longer (100 epochs as opposed to 15 epochs), the precision and recall values would be similar to the teacher's precision and recall values (which can be seen in milestone-3 of this project)