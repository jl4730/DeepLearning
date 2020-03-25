# DeepLearning and CNN
Summarize what I learned through Udacity

### 1.1 SimplePerceptronAlgorithm.py
Classify a group of points with y is either 0 or 1 using perceptron algorithm

### 1.2 GradientDescent.ipynb
Implement the gradient descent algorithm on the dataset with two classes

### 1.3 StudentAdmissions.ipynb
Put neural networks in practice by analyzing a dataset of student admissions at UCLA

### 1.4 StudentAdmissionsKeras
Use Keras to train a simple model

### 1.5 IMDB_In_Keras.ipynb
Using Keras to analyze IMDB Movie Data

### 1.6 Image Classifier Project
Develop code for an image classifier built with PyTorch

### 1.7 Image Classifier Project (Command Line)
Convert 1.6 into a command line application

Train a new network on a data set with train.py

Basic usage: python train.py data_directory
Prints out training loss, validation loss, and validation accuracy as the network trains

Options:
Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
Choose architecture: python train.py data_dir --arch "vgg13"
Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
Use GPU for training: python train.py data_dir --gpu

Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

Basic usage: python predict.py /path/to/image checkpoint

Options:
Return top KK most likely classes: python predict.py input checkpoint --top_k 3
Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
Use GPU for inference: python predict.py input checkpoint --gpu
