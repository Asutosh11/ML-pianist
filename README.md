<b>An experiment with generating music using Tensorflow/Keras</b>

For those who are wondering, what exactly machine learning is, its basically a set of mathematical formulae/techniques
where you train an anlgorithm to establish a relationship between input and output data or when given a lot of input data, 
it creates gropus of similar data. You either predict something given an input data or classify data into groups.

Some examples of these mathematical techniques are Linear Regression, SVM, Decision Trees, etc, 

Another hot stuff is Deep Learning. Deep Learning is a subset of machine learning. The way it differs from traditional
Machine Learning techniques is, Deep Learning algorithms are designed as per human brain and are called Neural Networks.

There are many types of Neural Networks. ANN, CNN, RNN (LSTM, GRU), GAN, etc.

Out of these, an RNN is what I have choosen to generate music. But why? RNNs are used for time series analysis. Problems where 
every output is dependant on a series of previous inputs and not just last one value. 
For example, every musical note is preceeded by a series of notes. The price of a stock fo any company follows a trend 
that is preceeded by a series of stock prices, etc

I have implemented this using Temsorflow/Keras Deep Learning framework. 
The reason is Tensorflow offers a bridge to convert the models to tflite format, to run your models on Android/iOS as well and I would be needing that for deploying this on an Android device. 
