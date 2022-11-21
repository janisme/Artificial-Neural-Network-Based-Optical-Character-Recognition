# Artificial-Neural-Network-Based-Optical-Character-Recognition



## 1	Introduction

OCR stands for Optical Character Recognition, which is a technology to recognise characters include numbers, letters.. etc from images, such as handwriting, scanned documents and photos.   
Neural networks have great application in data mining used in sectors. For example economics, for pattern recognition, etc. It could be also used for data classification in a large amount of data after careful training.With the help of the combination ANN and OCR, people no longer need to manually retype important documents when entering them into electronic databases. Instead, OCR extracts relevant information and enters it automatically. The result is accurate, the processing time is less.  
## 2	Image Datasets
In this project, we use a dataset of images named “ETL Character Database” that is a collection of images of about 1.2 million hand-written and machine-printed numerals, symbols, Latin alphabets and Japanese characters and compiled in 9 datasets (ETL-1 to ETL-9). This database has been collected by Electrotechnical Laboratory (currently reorganized as the National Institute of Advanced Industrial Science and Technology (AIST)) under the cooperation with Japan Electronic Industry Developement Association (currently reorganized as Japan Electronics and Information Technology Industries Association), universities and other research organizations for character recognition researches from 1973 to 1984.  
<img width="366" alt="image" src="https://user-images.githubusercontent.com/49497111/203133731-5bd263c4-a024-4429-a430-5c4d9c6e8818.png">.  
Figure 2-1:ETL-1 datasets.   
 <img width="330" alt="image" src="https://user-images.githubusercontent.com/49497111/203133755-e12e6a58-d980-4fa7-8503-0f52f01be9f9.png">. 
Figure 2-2: ETL-1 Description.   

In the scope of the project, we use 1000 images of each number from 0 to 9 which are extracted from ETL1C-01 dataset and rename each image by the format I.png. (For example, the images with names I2001.png to I3000.png will be the number 2) every images of character are in the same size 63*64 pixels.
 
## 3	Image Pre-processing
* 3.1	Image Acquisition and Binarization. 
* 3.2	Normalization. 
* 3.3	Whitening.    
* 3.4	Remove small objects
## 4	Feature Extraction
* 4.1	 Closed area       
* 4.2	  Walsh-Hadamard transform. 
* 4.3	Horizontal Symmetry, Vertical Symmetry. 
* 4.4	Sum of pixel at H30, H50, H80, V30, V50, V80. 
## 5	Artificial Neural Networks 
>An Artificial Neural Network (ANN) is a mathematical model that tries to simulate the structure and functionalities of biological neural networks. Basic building block of every artificial neural network is artificial neuron, that is, a simple mathematical model (function). 
Such a model has three simple sets of rules: multiplication, summation and activation. At the entrance of artificial neuron the inputs are weighted what means that every input value is multiplied with individual weight. In the middle section of artificial neuron is sum function that sums all weighted inputs and bias. At the exit of artificial neuron the sum of previously weighted inputs and bias is passing trough activation function that is also called transfer function 
 <img width="454" alt="image" src="https://user-images.githubusercontent.com/49497111/203135261-f14d8a07-4eb3-4ce2-8f55-f3563a770e78.png">
Figure 5-1: Working principle of an artificial neuron.   

### 5.1	Artificial neuron 
>Artificial  neuron  is  a  basic  building  block  of  every  artificial  neural  network.  Its  design  and  functionalities  are  derived  from  observation  of  a  biological  neuron  that  is  basic  building  block  of  biological  neural  networks  (systems)  which  includes  the  brain,  spinal  cord  and  peripheral ganglia.   
>Similarities in design and functionalities can be seen in figure where the left  side  of  a  figure  represents  a  biological  neuron  with  its  soma,  dendrites  and  axon  and  where  the  right  side  of  a  figure  represents  an  artificial  neuron  with  its  inputs,  weights,  transfer function, bias and outputs.  
 <img width="454" alt="image" src="https://user-images.githubusercontent.com/49497111/203135358-986bf26b-28f5-4a9d-aa07-9412af185db8.png">
Figure 5-3: Biological and artificial neuron design. 
In biological neuron information comes into the neuron via dendrite, soma processes the information and passes it on via axon. 
With artificial neuron, the information comes into the body of an artificial neuron via inputs that are weighted (each input can be individually multiplied with a weight). The body of an artificial neuron then sums the weighted inputs, bias and “processes” the sum with a transfer function. At the end an artificial neuron passes the processed information via output(s). Benefit of artificial neuron model simplicity can be seen in its mathematical description below: 
<img width="250" alt="image" src="https://user-images.githubusercontent.com/49497111/203135414-77f2c8a7-0aaa-4de1-9dc1-b6858d6787de.png">
<img width="381" alt="image" src="https://user-images.githubusercontent.com/49497111/203135433-b45c39c9-1180-424b-a460-05325b0c3b91.png">. 

Transfer function defines the properties of artificial neuron and can be any mathematical function. In most cases we choose it from the following set of functions: Step function, Linear function and Non-linear (Sigmoid) function.
Step function is binary function that has only two possible output values (e.g. zero and one). That means if input value meets specific threshold the output value results in one value and if specific threshold is not meet that results in different output value. 

 <img width="220" alt="image" src="https://user-images.githubusercontent.com/49497111/203135669-924a418f-0644-4bc3-87ec-232d52ba5f09.png">. 
 
When this type of transfer function is used in artificial neuron we call this artificial neuron perceptron. Perceptron is used for solving classification problems and as such it can be most commonly found in the last layer of artificial neural networks. 
In case of linear transfer function artificial neuron is doing simple linear transformation over the sum of weighted inputs and bias. Such an artificial neuron is in contrast to perceptron most commonly used in the input layer of artificial neural networks. 
When we use non-linear function the sigmoid function is the most commonly used. Sigmoid function has easily calculated derivate, which can be important when calculating weight updates in the artificial neural network. 

## 6	Convolution Neural Network Applied On OCR 
### 6.1	Introduction
CNN is a supervised learning and a subclass if ANN1, the different between convolution network is that it uses at least one layer one convolution method to process data.
A convolution unit receives its input from multiple units from the previous layer which together create a proximity. Therefore, the input units share their weights, in another way is all the points in one feature map shares the same filter.
And by choosing the CNN instead of ANN, we can get the following benefits:
* 1.	Many-to-one mappings(pooling) : In this way we can use fewer parameter, and it will reduce the complexity(points) and also the chance of overfitting.
* 2.	Locally connected networks: Convolution has the property to combine neighbor points, therefore, when applied to pictures it shows the essence of the graph more precisely.
* 3.	Filter: Diminishing the noise and extract the character of picture.  

 <img width="359" alt="image" src="https://user-images.githubusercontent.com/49497111/203135767-96a79215-db92-4f6e-aee3-5614a772cc10.png">
Figure 6-1: The concept of CNN. 

### 6.2	Procedure of CNN
The essential layers in a convolution network has convolution layer, pooling layer, flatten layer and hidden layer.
* Convolution layer:
The concept of this is to put the original matrix through a filter matrix, it can extract the feature of the original picture. The purpose of this procedure is to make a feature map that can shows the different feature of each graph.
<img width="209" alt="image" src="https://user-images.githubusercontent.com/49497111/203135935-a0c65cb4-431a-4d15-9515-fba0a0e48175.png">
Figure 6-2: Mathematically explanation. 

From the left side of the matrix, multiply the blue part of first matrix with the colorful filter, and put it in the relative position in the feature map. The left side of pictures visualize the procedure of the convolution.
And to the notice, since the graph we show is the original graph with the initial random filter matrix, the feature map on the top still can be recognize, but with more and more numbers and interactions of optimization for the filter, we will not be able to recognized the feature map at the end.

<img width="290" alt="image" src="https://user-images.githubusercontent.com/49497111/203136025-e687fc48-a63f-44b9-b4cd-32f193e866d4.png">
<img width="290" alt="image" src="https://user-images.githubusercontent.com/49497111/203136032-dc1e3493-8c59-49c0-8ec2-776c95efc872.png"> 
Figure 6-4: Demo from our dataset with one convolution and two different activation function. 

* Pooling Layer: 
	In general, CNN use the max pooling function to extract the max number in the range, and in our project, there is no exception. 
	The propose of the pooling is to downsize the points to lower the memory used and the complexity and it is emphasized on whether the graph has the feature than where is the feature.
	We use 2*2 range pooling to downsize the points in the graph. 
  
<img width="263" alt="image" src="https://user-images.githubusercontent.com/49497111/203136067-b228ee2f-a564-4f27-9b70-a7bb233dd678.png">
<img width="307" alt="image" src="https://user-images.githubusercontent.com/49497111/203136077-7ffef6aa-f9f5-47de-bfec-1540f0859bf3.png">
<img width="307" alt="image" src="https://user-images.githubusercontent.com/49497111/203136084-a56430cb-5079-4cc7-9022-03fae0edee2b.png">
Figure 6-5: Demo from our data with 2*2 max pooling function. 

* Flatten layer
	It is a preparation for the next level of analysis, we flatten the 2D table into 1D column, so that we can start to do the ANN(here we call it hidden layer analysis).
  
 <img width="246" alt="image" src="https://user-images.githubusercontent.com/49497111/203136097-92232ceb-4c26-4376-819d-d64a27379bce.png">. 
 
Figure 6-6: Flatten. 
  
* Dropout layer
	The purpose of the dropout is to reduce the probability of overfitting during the training session. By setting the dropout percentage (0-1) the optimization process will randomly neglect the percentage of the weight. Rather delete it, the weight still would be saved for the prediction.  
  
<img width="169" alt="image" src="https://user-images.githubusercontent.com/49497111/203136222-176c51b2-4f6d-4fda-b301-d38e4930c3e3.png">. 

Figure 6-7: Dropout. 
  
* Hidden layer
	It’s an ANN layer that links the previous points to the next layer’s with the parameter we called ‘weight’. And it’s also the last layer of the function, therefore the final outpoint will have the same number like the class of classification.  
  
  <img width="234" alt="image" src="https://user-images.githubusercontent.com/49497111/203136252-fca252cf-5a49-4071-9e93-2b809c6ed584.png">. 
  
Figure 6-8: Hidden layer to predict output. 

## 6.3	Model 
In our project we build the model with two convolution layer and two hidden layers.  

<img width="278" alt="image" src="https://user-images.githubusercontent.com/49497111/203136317-b28697eb-8707-4a34-bf83-c70ffbbc8882.png">.  

Figure 6-9: Model.   
* As you can see the data set is a 63*64 graph. 
* At the first two convolution layers, we each use 16 and 36 filters with the 5*5 pooling mask.
* In the middle, we use two dropout layers to reduce the overfitting.
* In the last part we use two hidden layers with 128 and 10 neural each.
* The total parameter in this model is 1,122,190. 

## 6.4	Results
* The training sets has 8000 data and the testing sets has 2000 data.
* We use 7 interactions each with batch size of 300.
* Each interaction took about 40 seconds
* In the end we get 96.5% of accuracy.  

<img width="334" alt="image" src="https://user-images.githubusercontent.com/49497111/203136670-580604b2-bd87-4c36-9249-5b56efe680ed.png">. 
<img width="203" alt="image" src="https://user-images.githubusercontent.com/49497111/203136683-53e37cb8-7084-45b6-95fa-8e545816c127.png">. 

* Prediction 
By using to the prediction, we get 96% of accuracy. And the mostly confused number are 8-6 and 9-7. We supposed that is because the dataset we use is the hand writing numbers, so it is also confusing by the human eyes.





