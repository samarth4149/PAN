from .initializations import *
import tensorflow as tf

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    # Properties
        name: String, defines the variable scope of the layer.
            Layers with common name share variables. (TODO)
        logging: Boolean, switches Tensorflow histogram logging on/off
    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, input):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/input', input)
            outputs = self._call(input)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """
    Dense layer (with dropout):
        output = act(input*weight + bias)
    """
    def __init__(self, input_dim, output_dim, is_train, dropout=0., act=tf.nn.relu,
                 bias=False, batch_norm=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_random_uniform(input_dim, output_dim, name="weights")

            if bias:
                self.vars['node_bias'] = bias_variable_zero([output_dim], name="bias_n")


        self.bias = bias
        self.batch_norm = batch_norm
        self.is_train = is_train

        self.dropout = dropout
        self.act = act
        if self.logging:
            self._log_vars()

    def _call(self, input):
        x_n = input
        x_n = tf.nn.dropout(x_n, 1 - self.dropout)#drop out layer
        x_n = tf.matmul(x_n, self.vars['weights'])

        if self.bias and not self.batch_norm: # do not use bias if using bn
            x_n += self.vars['node_bias']
        # above step: x_n = x_n*Weights + bias

        n_outputs = self.act(x_n)#activation function

        if self.batch_norm:
            n_outputs = tf.layers.batch_normalization(n_outputs, training=self.is_train)

        return n_outputs

    def __call__(self, input):
        with tf.name_scope(self.name):
            if self.logging:
                tf.summary.histogram(self.name + '/input', input)
            outputs_n = self._call(input)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs_n', outputs_n)
            return outputs_n


class GCN(Layer):
    """
    Graph convolution layer for multiple degree adjacencies
    
    
    
    
    """
    def __init__(self, input_dim, output_dim, support, num_support, is_train, dropout=0.,
                 act=tf.nn.relu, bias=False, batch_norm=False, init='def', **kwargs):
        super(GCN, self).__init__(**kwargs)
        assert init in ['def', 'he']
        with tf.variable_scope(self.name + '_vars'):
            if init == 'def':
                init_func = weight_variable_random_uniform
            else:
                init_func = weight_variable_he_init

            
            self.vars['weights'] = [init_func(input_dim, output_dim,
                                            name='weights_n_%d' % i)
                                            for i in range(num_support)]

            if bias:
                self.vars['bias_n'] = bias_variable_zero([output_dim], name="bias_n")
                
            self.weights = self.vars['weights']

        self.dropout = dropout

        self.batch_norm = batch_norm
        self.is_train = is_train

        self.bias = bias
        # TODO, REMOVE
        # support = tf.sparse_split(axis=1, num_split=num_support, sp_input=support)
        self.support = support

        self.act = act

        if self.logging:
            self._log_vars()

    def _call(self, input):
        x_n = tf.nn.dropout(input, 1 - self.dropout)#x_n with dropout

        supports_n = []

        for i in range(len(self.support)):#K adjency nodes
            wn = self.weights[i]
            
            tmp_n = dot(x_n, wn, sparse=self.sparse_inputs)
            #multiply x_n with its weights, z_i*theta_0 in equation (1)

            support = self.support[i]

            # then multiply with rating matrices
            supports_n.append(tf.sparse_tensor_dense_matmul(support, tmp_n))
            #dense matrix tmp_n multiply spare matrix support(adjency matrix)

        z_n = tf.add_n(supports_n)#add all the elements of 'supports_n' together 

        if self.bias:
            z_n = tf.nn.bias_add(z_n, self.vars['bias_n'])

        n_outputs = self.act(z_n)

        if self.batch_norm:
            n_outputs = tf.layers.batch_normalization(n_outputs, training=self.is_train)

        return n_outputs

    def __call__(self, input):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/input', input)
            outputs_n = self._call(input)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs_n', outputs_n)
            return outputs_n


class MLPDecoder(Layer):
    """
    MLP-based decoder model layer for edge-prediction:
        p = sigmoid(|h_i-h_j|*weights + bias)
    r_indices:      placeholders['row_indices']: (685024,) positive/negative edge indices
    c_indices:      placeholders['col_indices']: (685024,) positive/negative edge indices
    """
    def __init__(self, num_classes, r_indices, c_indices, input_dim,
                 dropout=0., act=lambda x: x, n_out=1, use_bias=False,num_attribute=10,
                 learnedmask=False, mask=False,**kwargs):
        super(MLPDecoder, self).__init__(**kwargs)
        self.option = 1 # 1: difference, 2: pointwise 3: concatenation
        self.normalize=True

        if self.option==3:
            input_dim = 2*input_dim
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_random_uniform(input_dim, n_out, name='weights')
            self.vars['weights_2'] = weight_variable_random_uniform(input_dim,n_out,name='weights_2')
            if use_bias:
                self.vars['attribute_bias'] = bias_variable_zero([n_out], name="bias")
                self.vars["attention_bias"] = bias_variable_zero([n_out],name="attention_bias")
                self.vars["score_bias"] = bias_variable_zero([n_out],name='score_bias')
        
        self.input_dim = input_dim
        self.num_attribute = num_attribute
        self.learnedmask = learnedmask
        self.mask = mask
        
        self.r_indices = r_indices
        self.c_indices = c_indices

        self.dropout = dropout
        self.act = act
        self.n_out = n_out
        self.use_bias = use_bias
        if self.logging:
            self._log_vars()
            
    
    def _call(self, inputs):
        node_inputs = tf.nn.dropout(inputs, 1 - self.dropout)
        
        # r corresponds to the selected rows, and c to the selected columns
        row_inputs = tf.gather(node_inputs, self.r_indices)#(None,200)
        #tf.gather: use r_indices to extract the retrievaled rows
        col_inputs = tf.gather(node_inputs, self.c_indices)
        # (None,350), for polyvore outfit dataset, None=685024
       
        #option 1: absolute difference
        if self.option==1:
            diff = tf.abs(row_inputs - col_inputs)
        #option 2: element-wise multiplication
        elif self.option==2:
            if self.normalize==True:
                row_inputs = tf.math.l2_normalize(row_inputs,axis=1);
                col_inputs = tf.math.l2_normalize(col_inputs,axis=1);
                print(row_inputs.shape)
                print(col_inputs.shape)
            diff = row_inputs*col_inputs# pointwise multiplication
            print(diff.shape)
        #option 3: concatenate features
        elif self.option==3:
            diff = tf.concat([row_inputs,col_inputs],axis=1)
        
        attributes = tf.matmul(diff, self.vars['weights'])#(None,20)
        if self.use_bias:
            attributes += self.vars['attribute_bias']
        sigmoid_attributes = tf.nn.sigmoid(attributes)    
        
        if self.n_out == 1:
            print("number of output nodes: 1")
            outputs = tf.squeeze(sigmoid_attributes) # remove single dimension
            return outputs
        else:
            print("multiple output nodes")
            output2 = tf.nn.softmax(tf.matmul(diff,self.vars['weights_2']))#(None,n_output)
            outputs = tf.reduce_sum(sigmoid_attributes*output2,1)

        outputs = self.act(outputs)
        outputs = tf.reshape(outputs,[-1,1])
        return tf.concat([tf.clip_by_value(outputs,1e-10,1-1e-10),tf.clip_by_value(sigmoid_attributes,1e-10,1-1e-10),output2],1)
        

