# used for model 2
# import the dataset 
from dataset.data import *

# instantiate a class for the USE (Universal Sentence Encoder) model
import keras 

# to import the model 
import tensorflow_hub as hub 

# inherith from the keras Layer 
class USEModel(keras.Layer):
    # initialize the class
    def __init__(self, **kwargs):
        # import the inherited methods from keras Layer
        super().__init__(**kwargs)

        self.use_layer = hub.KerasLayer(
            handle='https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/2',
            input_shape=[],
            dtype=tf.string,
            name='Universal_Sentence_Encoder'

        )

    # pass the inputs to the layer (call method in Neural Networks)
    def call(self, inputs):
        return self.use_layer(inputs)





    




