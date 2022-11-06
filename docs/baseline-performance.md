# Baseline Performance

## UNet Explanation
UNet is a deep learning architecture that is often used for its performance in image recognition.  It is a neural network, and it possesses qualities that other neural networks have--an input layer, and an output layer.  The output layer ultimately decides the outcome, and therefore must be completely connected with the preceding layer's nodes.  However, the hidden layers of UNet are quite different from other neural networks in the sense that they have a set of filters.  These filters are responsible for recognizing certain details and patterns of an image.  For example, in the first convolutional layer, one of its filters can be responsible for detecing edge patterns, which is the more simple yet fundamental steps in image recognition.  Essentially, UNet is a model for classification problems involving images. 

### Convolutional Layer Filters
Images are made up of many different details--layers, color, gradient, texture, and more.  These details give way to many possible patterns that can be recognized by machines.  The point of filters in convolutional layers is to detect these patterns.  In UNet, filters start out to be very basic, like detecting the edges within an image.  The next filters shrink to detect more complex patterns--in the case of this project, filters will be able to detect buildings, land, roads, vegetation, and water.  Anything that is not one of those five will be marked as unlabeled. <br>

Technically, a filter can be seen as a rather small matrix that will be slid across the pixels of the image.  In the case of this project, the dimensions of this filter is 3 by 3 in the contraction path layers.  The values of this filter are initially randomized.  As the 3 by 3 filter is slid across the 3 by 3 pixel sets of the image, the dot product of the two matrices are calculated and stored in another matrix that holds these dot products.  This process is repeated for however many filters each layer has, and the output of the layer (the matrix of dot products) will be the input of the nex convolutional layer.

### Max Pooling
In the case of this project, max pooling is an operation performed after obtaining an output from a convolutional layer.  Max pooling is an operation in which in a "pool" of values (pixel set of size 2 by 2 in this case), the maximum value of that pool is obtained and inserted into an output matrix.  This 2 by 2 pixel filter is slid across the image according to a stride (unspecified in the unet model code) until all maximum values are obtained from each pool.  The resulting matrix is a shrunken matrix consisting of the maximum values of each pool of pixels taken from the preceding matrix.<br>

There are a couple of reasons to use max pooling.  The first being that shrinking the data set containing the image information will result in lower computational cost down the line.  It is important to note that there are several convolutional layers in a UNet--in the case of this project, the contraction path has 5 connvolutional layers.  While the image resolution is already lowers the resolution through the filtering processes, the resulting image may still be quite larger.  Max pool further lowers this resolution, while also preserving the more impactful pixels in each pool of pixels.<br>

The second reason for using max pooling is to reduce overfitting.  It will be further explained later in this document that overfitting is still an issue with the current state of the program, but max pooling tries to prevent it to some extent.  Max pooling focuses on the most impactful values in a given pool.  Therefore, while there is less data for the model to train with (this prevents overfitting), the provided data consists of the most activated values.

### Encoding and Decoding at a High Level
There are essentially two parts to the UNet model--the encoder and decoder.  In the simple_multi_unet_model.py file, the encoder part is commented as the contraction path.  This label suits it, as the image contracts as it moves through the convolutional layers--the resolution of the image is lessened to focus on the the more defining details rather than the region in which that details appears at in the picture.  Looking at the code, it can be seen that the depth of the contraction path increases as follows: 16=>32=>64=>128=>256.<br>

The decoder is commented as the expansive path.  Again, the name depicts its operation--re-expanding the image to a larger size after analyzing the fine details.  The depth decreases as follows (due to the re-expanding of the image): 256=>128=>64=>32=>16.  The output of the convolutional transpose layers in the decoder side are concatenated with the feature maps from the encoder side--this increases the accuracy of determining locations of certain classified objects in the image.  This concatenation is finally put through a last convolutional layer with a 1 by 1 filter--this result will be the outputs of the UNet model.

## Image Resizing and Dataset Preparation
The purpose of this milestone was to produce plots that show 10 segmented images, training and validation loss vs epoch graphs, and the precision-recall curve.  Firstly, the kaggle dataset needed to be downloaded.  This dataset includes 2 subdirectories: images and masks.  The first part of 228_training_aerial_imagery was designed to manipulate the images inside of these subdirectories and convert them into workable data for the python program.  IN our case, the root directory was the parent directory of the images and masks directory--Semantic segmentation dataset/.  The program also initializes a patch_size of 256, which means that the images will be split into patches of size 256x256x3 (3 different channels--RGB).  The following steps were taken for the populating the image_dataset array:
- find SIZE_X and SIZE_Y through integer dividing by the patch_size and multiplying by the patch size (if real size of the X axis is 2149, we must crop this image to make the size of the X axis 2048 since 2048 is divisible by 256)
- we crop the image based on SIZE_X and SIZE_Y--this will be the image we work with, as it now has dimensions divisible by our patch_size 256
- the image is turned back into a numpy array of dimensions SIZE_Y by SIZE_X by 3
- patchify serves to break up this large image into adjacent tiles of size 256 by 256 by 3
- for each patch, we must scale the rgb values with MinMaxScaler because rgb values can range from 0 to 255, where we want to restrict this to a probability (between 0 and 1)
- each scaled patch is added to the image_dataset, creating an array of numpy arrays

The following steps were taken for populating the masks_dataset array:
- essentially the same as the process done for the image_dataset array
- the only difference is converting the colors from bgr to rgb (openCV reads colors in bgr order)

After performing the above steps, the result is 2 numpy arrays, each containing image patches of size 256 by 256 by 3 in the form of numpy arrays.<br><br>

Recall that the  images (and patches) are ran through 3 color channels--rgb.  However, in the kaggle data set, we are given the hexadecimal codes for the 6 identifiable classes: building, land, road, vegetation, water, and unlabeled.  Therefore the given hexadecimal values must be converted to rgb values.<br><br>

The mask_dataset (which will be used for training) must also be converted to a simpler array that does not include 3 separate channels for rgb.  To achieve this, the rgb values representing each class are mapped to a certain classification number:
- Building = 0
- Land = 1
- Road = 2
- Vegetation = 3
- Water = 4
- Unlabeled = 5

Given these values, the mask_dataset is looped through to convert the rgb to one of the above listed classification numbers and append it to a new numpy array named labels.<br><br>

Note that the previous step gives each pixel in the data set a value between 0 and 5 inclusive.  However, this is not categorical, meaning a pixel being part of a road is not determined with a simple yes or no.  Therefore, the dataset is split over the 6 classes--building, land, road, vegetation, water, and unlabeled.  This is similar to the kaggle dataset for the Logistic Regression homework assignment, where there were several features as columns that held a value 0 or 1.