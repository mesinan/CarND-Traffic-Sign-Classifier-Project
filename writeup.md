# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/mesinan/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.


I have included one sample traffic sign following with overall bar chart showing how the labels are distributed in training set.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

In my first desing, I tried to go without converting grayscale.

I set up a model based on LeNet and started from original model.
My first attempt was around accuracy of 0.800 , it was far less from expected.
I first played with epochs and batch size. increasing batch size decreased the accuracy. 
Increasing epochs does not effected after 30 epochs.

Then I started to modify the size of the layers of the model. 

I increased sizes of all layers, it increased the accuracy but couldn't exceed 0.920.
Then I tought that maybe I should start trying grayscale.
Grayscale worked. In less than 10 Epochs, accuracy increased to 0.930 and more.
Then I decreased the first layer size, because I tought that biggest distinction between signs would be done in higher level layers. 
As I guessed decreasing size of first convolution layer did not have any negative affect.
At last step I snipped 5 signs from google street images around Berlin.
After I run my network, I saw that accuracy was around 0.600, far less than expected again.
But when I plotted the result of each sign afterwards, I saw that all leading probabilities are correct for signs.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x36 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x36 				|
| Convolution 3x3	    | 1x1 stride,  outputs 10x10x108				|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x108					|
| Fully connected		| outputs 2700 									|
| Fully connected		| outputs 300 									|
| RELU					|												|
| Fully connected		| outputs 160 									|
| RELU					|												|
| Fully connected		| outputs 43 									|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I tried to decrease learning rate by multiplying .1 but it does not have any visible positive affect so I remain it as 0.001
I used adamoptimizer as discussed in LeNet example.
I choose epochs as 10 and it seems enough, this is great.
Increasing batch size had negative effect, Then I used it as 64.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.944
* validation set accuracy of "none"
* test set accuracy of 0.800

If an iterative approach was chosen:
* I choose LeNet because I tought that it is simple enough to understand and modify. If it wouldn't work, I would try AlexNet but it worked.
* When I first transfer the model, layers were too small. After grayscale conversion and layer size extension, it worked well.
* I did not add or remove any new layer to existing architecture, I only changed the sizes of layers. 
* For convolutions and fully connected layers, I changed shapes of Weights and length of bias arrays. I did not make any changes on pooling layers. 
* I think dropout may work well since there are a lot of repetetive data, and may prevent overfitting.

If a well known architecture was chosen:
* LeNet
* LeNet is an improved model for this kind of simple datasets.
* Result of 5 test signs show that my modified LeNet model choose all af the sign labels correctly
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Images are listed in ipynb file. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No Entry      		| No Entry   									| 
| Speed Limit 50 Kmh 	| Speed Limit 50 Kmh 							|
| Yield					| Yield											|
| Turn Right Ahead 		| Turn Right Ahead				 				|
| Bumpy Road			| Bumpy Road	      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 0.938

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 14th cell of the Ipython notebook.

Most of the images are predicted with probabality of one except Speed Limit 50kmh.
Speed limit 50 kmh has some probabality of being speed limit 30 kmh and speed limit 80 kmh

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| No Entry   									| 
| 1     				| Turn Right Ahead										|
| 1						| Yield											|
| 1		      			| Bumpy Road					 				|
| .8				    | Speed Limit 50kmh    							|

