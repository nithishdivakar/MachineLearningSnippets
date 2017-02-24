from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class ListaLayer(Layer):
    def __init__(self,
                 #input_dim, 
                 unroll_steps,
                 dictionary_size, 
                 **kwargs
    ):
        self.unroll_steps = unroll_steps
        self.dictionary_size = dictionary_size
        super(ListaLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        def scalar(shape,name=None):
            return 5.0*K.ones(shape,name=name)
        
        assert len(input_shape) == 2
       
        self.W = self.add_weight(
                    shape=(input_shape[1], self.dictionary_size),
                    initializer='glorot_uniform',
                    trainable=True,
                    name="W"
                )
        
        self.Theta = self.add_weight(
                    shape=(self.dictionary_size,),
                    initializer='glorot_uniform',#'one',
                    trainable=True,
                    name="Theta"
                )
        
        self.S = self.add_weight(
                    shape=(self.dictionary_size,self.dictionary_size),
                    initializer='glorot_uniform',
                    trainable=True,
                    name="S"
                )
        
        self.Dx = self.add_weight(
                    shape=(self.dictionary_size,input_shape[1]),
                    initializer='glorot_uniform',
                    trainable=True,
                    name="Dx"
                )
        
        self.built = True
                          
    def call(self, y, mask=None):
        '''
        Theta is always positive. To enforce that, here we simply add a relu 
        activation on all itss usage. More over, to prevent any division overflow, 
        all usage of Theta is also added by a small epsilon
        '''
        def unit_threshold(v):
            return K.sign(v)*K.relu(K.abs(v) - 1.0) # clipping negative values.
        
        def F(u):
            u = unit_threshold(u)
            u = u * (K.relu(self.Theta)+K.epsilon())
            u = K.dot(u,self.S)
            u = u / (K.relu(self.Theta)+K.epsilon())
            return u   
        
        
        xW = K.dot(y, self.W) / (K.relu(self.Theta)+K.epsilon())
        
        y = xW
        z = xW
        for i in range(self.unroll_steps):    
            z = y+F(z)
            
        a = z
        a = unit_threshold(a)
        a = a * (K.relu(self.Theta)+K.epsilon())
        x = K.dot(a,self.Dx)
        return x
   
    def get_output_shape_for(self, input_shape):
        return input_shape
