# Handwritten-Equation-Solver

A Handwritten Equation Solver built using Convolutional Neural Network and Optical Character Recognition. CNN model is used for recognition of digits and symbols. OCR is used for processing the the image and segmentation.

The model is capable of solving equations of any degree (linear, quadratic, cubic and so on). The variable should always be 'x' as it's trained for images of x alphabet only.

An accuracy of **0.9827** is achieved with loss function as 0.0656 for recognition


![ezgif com-gif-maker](https://user-images.githubusercontent.com/59358031/124664638-61785180-dec9-11eb-8ea8-0cd4f5a73ebe.gif)

# Dataset Description
The dataset is downloaded from Kaggle “Handwritten Math Symbol Dataset”.The dataset consists of over 1,00,000 images of many Mathematical symbols and digits. Only 0 to 9 digits  ‘+’, ‘-’ and ‘x’ are used in the project which estimates around 47,000 images. Each image is of size 45x45 and the image format id JPG.

The Dataset can be downloaded from https://www.kaggle.com/xainano/handwrittenmathsymbols
# Requirements 

* Python 3  https://www.python.org/downloads/

* Anaconda   https://www.anaconda.com/download/

* Keras  
```                                                                                                          
pip install keras
```

* OpenCV
```
pip install opencv-python
```

* Pillow
```
pip install pillow
```
* Sympy
```
pip install sympy
```

# How to run

Clone the repository using 
```
https://github.com/prateeek1/Handwritten-Equation-Solver.git
```

You can run all the three ipynb files either separately or sequentially.

* For running LoadingDataset.ipynb first download train the dataset from the above given link and extract it in the folder containing LoadingDataset.ipynb file.

* For running CNNModel.ipynb, you either need to download train_final.csv or you can run it after succesfully running LoadingDataset.ipynb.

* For running EquationSolver.ipynb, you either need to download model folder or you can run it after succesfully running CNNModel.ipynb file.

# Convolutional Neural Network

![image](https://user-images.githubusercontent.com/59358031/124674862-3138af00-ded9-11eb-99dd-1b68b2c51585.png)
   
     
* A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other.

* Every CNN architecture is provided with a fully connected layer in the end where learning of non-linear combination of high-level features takes place.
* In our case, we must convert the input image matrix into a compatible form to make it suited for multi-layer perceptron. Therefore, we flatten the matrix and produce a single vector which is fully connected. This flattened data is then forwarded to a feed forward neural network. At every iteration of the training dataset, back-propagation comes into play and helps in adjustment of the weights of the inputs.
Convolution Neural Networks
* After running the entire set of iterations for a specified number of epochs, the model becomes able in distinguishing between the important and the trivial features of an image. In the end since this is multiclass classification problem, we apply a “Softmax” function to convert the vector values into a probability distribution consisting of the same number of probabilities as the elements in the vector.


* ###  Components of Convolutional Neural Network
* Two Convolutional Layers

* Two MaxPooling Layers

* Dropout Layer

* Flatten Layer

* Two Dense Layers using Relu

* Output Layer using Softmax

* Adam Optimizer



* ### Model Summary

![jp](https://user-images.githubusercontent.com/59358031/124675569-85905e80-deda-11eb-95e5-88c28c35bf15.png)

* The model summary shows that it is a sequential model the first convolutional layer outputs an image of size 24x24 which is downsized by the max pooling layer to output 12x12 image which is passed onto the second convolutional layer and again pooling layer to output an image of 5x5 which is the flattened and at last dense layer is used to obtain an probability array of the 13 features (0 to 9 ,+,-,x).


# Image Pre-Processing


Pre-processing  of  the  input  image  is  the  procedure  which encompasses changes and modifications to the image to make it fit  for recognition.  The following  techniques may be used for image enhancement.  

* **Conversion of RGB to Gray-Scale** : First of  all this  coloured image is transformed into a  typical gray-scale  image and  is represented  through a  single matrix because the   detection  of characters  on a  coloured image  is more  challenging  than  on  a  gray-scale  image.
  
* **Binarization** :Binarization is the procedure of choosing a threshold value is for adaptation of pixel values into 0’s and 1’s. In This research for  horizontal  projection  calculation  1’s  represents  the  black pixels  while  0’s  characterizes the  white pixels. The threshold choice of binarization can approve two ways: overall threshold and  partial  threshold. Here Binary method is used.

*  **Segmentation** :Once the pre-processing of the input images is completed, sub-images of individual digits are formed from the sequence of images. Pre-processed digit images are segmented into a sub-image of individual digits, which are assigned a number to each digit. Each individual digit is resized into pixels. In this step an edge detection technique is being used for segmentation of dataset images.

*  **Feature Extraction**: After the completion of pre-processing stage and segmentation stage, the pre-processed images are represented in the form of a matrix which contains pixels of the images that are of very large size. In this way it will be valuable to represent the digits in the images which contain the necessary information. This activity is called feature extraction. In the feature extraction stage redundancy from the data is removed.

*  **Recognition**: Extracted feature vectors are then given input to the model which predicts the values of all the digits and variables in the equation and append them in the list of strings. This list is then converted into string.


# REFERENCE

1.	[A Comprehensive Tutorial to learn Convolutional Neural Networks from Scratch](https://www.analyticsvidhya.com/blog/2018/12/guide-convolutional-neural-network-cnn/)
2.	[Handwriting Recognition using CNN](https://aihubprojects.com/handwriting-recognition-using-cnn-ai-projects/)
3.	[Handwritten Digit Recognition GUI App](https://medium.com/analytics-vidhya/handwritten-digit-recognition-gui-app-46e3d7b37287)
4.	[OpenCV Documentation](https://www.docs.opencv.org/master/)




