from mynn.layers.conv import conv
from mynn.layers.dense import dense
from mygrad.nnet.initializers import glorot_uniform
from mygrad.nnet.activations import relu
from mygrad.nnet.layers import max_pool
from mygrad.nnet.layers import batchnorm
from mygrad.nnet.losses import softmax_crossentropy
import mygrad as mg
import numpy as np

class Model:
    def __init__(self, num_input_channels, f1, f2, f3, f4, d1, num_classes):
        """
        Parameters
        ----------
        num_input_channels : int
            The number of channels for a input datum
            
        f1 : int
            The number of filters in conv-layer 1
        
        f2 : int
            The number of filters in conv-layer 2

        f3 : int
            The number of filters in conv-layer 3

        f4 : int
            The number of filters in conv-layer 4

        d1 : int
            The number of neurons in dense-layer 1
        
        num_classes : int
            The number of classes predicted by the model.
        """
        # Initialize your two convolution layers and two dense layers each 
        # as class attributes using the functions imported from MyNN
        #
        # We will use `weight_initializer=glorot_uniform` for all 4 layers
        
        # Note that you will need to compute `input_size` for
        # dense layer 1 : the number of elements being produced by the preceding conv
        # layer
        # <COGINST>
        init_kwargs = {'gain': np.sqrt(2)}

        self.conv1 = conv(num_input_channels, f1, 4, 2, padding=(2, 1), 
                          weight_initializer=glorot_uniform, 
                          weight_kwargs=init_kwargs)
        self.conv2 = conv(f1, f2, 3, 1,
                          weight_initializer=glorot_uniform, 
                          weight_kwargs=init_kwargs)
        self.conv3 = conv(f2, f3, 2, 1, padding=(0,1),
                          weight_initializer=glorot_uniform, 
                          weight_kwargs=init_kwargs)
        self.conv4 = conv(f3, f4, 2, 1, padding=(0,1),
                          weight_initializer=glorot_uniform, 
                          weight_kwargs=init_kwargs)
        self.dense1 = dense(4284, d1, 
                            weight_initializer=glorot_uniform, 
                            weight_kwargs=init_kwargs)
        self.dense2 = dense(d1, num_classes, 
                            weight_initializer=glorot_uniform, 
                            weight_kwargs=init_kwargs)

    
    def __call__(self, x):
        """ Defines a forward pass of the model.
        
        Parameters
        ----------
        x : numpy.ndarray, shape=(N, 1, 32, 32)
            The input data, where N is the number of images.
            
        Returns
        -------
        mygrad.Tensor, shape=(N, num_classes)
            The class scores for each of the N images.
        """
        
        # Define the "forward pass" for this model based on the architecture detailed above.

        x = relu(self.conv1(x)[...,:-1,:])
        x = batchnorm(x, eps=1e-7)
        x = max_pool(x, (2, 2), 2)

        x = relu(self.conv2(x))
        x = batchnorm(x, eps=1e-7)
        x = max_pool(x, (2, 2), 2)

        x = relu(self.conv3(x)[...,:,:-1])
        x = batchnorm(x, eps=1e-7)
        x = max_pool(x, (2, 2), 2)

        x = relu(self.conv4(x)[...,:,:-1])
        x = batchnorm(x, eps=1e-7)
        x = max_pool(x, (2, 2), 2)

        x = relu(self.dense1(x.reshape(x.shape[0], -1)))
        return self.dense2(x)

    @property
    def parameters(self):
        """ A convenience function for getting all the parameters of our model. """
        params = []
        for layer in (self.conv1, self.conv2, self.conv3, self.conv4, self.dense1, self.dense2):
            params += list(layer.parameters)
        return params


    def accuracy(self, predictions, truth):
        """
        Returns the mean classification accuracy for a batch of predictions.
        
        Parameters
        ----------
        predictions : Union[numpy.ndarray, mg.Tensor], shape=(M, D)
            The scores for D classes, for a batch of M data points

        truth : numpy.ndarray, shape=(M,)
            The true labels for each datum in the batch: each label is an
            integer in [0, D)
        
        Returns
        -------
        float
            The fraction of predictions that indicated the correct class.
        """
        return np.mean(np.argmax(predictions, axis=1) == truth) # <COGLINE>

    def save_weights(self):
        """ 
        Saves the weights from the trained model

        Parameters
        ----------
        model : obj
            the trained model that is an instance of the Model class

        Returns
        -------
        str
            the file name in which the weights are stored
        """
        np.save("weights.npy", self.model.parameters)
        return "weights.npy"

    def load_weights(self, weights):
        """ 
        Loads the weights from the trained model

        Parameters
        ----------
        weights : str
            the file name in which the weights are stored

        Returns
        -------
        np.array
            loading in the saved weight matrix from the given file name
        """
        weight = np.load(weights)
        return weight