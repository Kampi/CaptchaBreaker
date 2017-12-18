from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense

class LeNet:
    
    def __init__(self):
        # Initialize the CNN model
        self.Model = Sequential()

    def build(self, width, height, depth, classes):
        # Create the input shape
        InputShape = (height, width, depth)
        self.Model.add(Conv2D(20, (5, 5), padding = "same", input_shape = InputShape))
        self.Model.add(Activation("relu"))
        self.Model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

        self.Model.add(Conv2D(50, (5, 5), padding = "same"))
        self.Model.add(Activation("relu"))
        self.Model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

        self.Model.add(Flatten())
        self.Model.add(Dense(500))
        self.Model.add(Activation("relu"))

        self.Model.add(Dense(classes))
        self.Model.add(Activation("softmax"))

        return self.Model