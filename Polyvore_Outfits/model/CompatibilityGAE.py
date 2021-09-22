from .layers import *

from .metrics import softmax_confusion_matrix, sigmoid_cross_entropy, softmax_cross_entropy
from .metrics import sigmoid_accuracy,binary_accuracy


flags = tf.app.flags
FLAGS = flags.FLAGS

num_sup = 206
lamda = 10

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'wd'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None
        self.output2 = None

        self.loss = 0
        self.total_loss = 0 # to use with weight decay
        self.accuracy = 0
        self.confmat = 0
        self.optimizer = None
        self.opt_op = None
        self.global_step = tf.Variable(0, trainable=False)
        if 'wd' in kwargs.keys():
            self.wd = kwargs.get('wd')
        else:
            self.wd = 0.

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        #activations: [(None,None),(None,200),(None,200),(None,1+n_output)]
        #self.outputs,self.output2 = tf.split(self.activations[-1], [1, int(self.activations[-1].shape[-1]-1)], 1)
        if self.n_out==1:
            self.outputs = self.activations[-1]
        else:
            self.outputs,self.output2,_ = tf.split(self.activations[-1],[1,num_sup,int(self.activations[-1].shape[-1]-num_sup-1)],1) 
            self.outputs = tf.reshape(self.outputs,[-1,])
        # Build metrics
        self._loss()
        self._accuracy()
        self._confmat()

        if self.wd:
            reg_weights = tf.get_collection("l2_regularize")
            loss_l2 = tf.add_n([ tf.nn.l2_loss(v) for v in reg_weights ]) * self.wd
            self.total_loss += self.loss + loss_l2
        else:
            self.total_loss = self.loss

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            reg_weights = tf.get_collection("l2_regularize")
            self.opt_op = self.optimizer.minimize(self.total_loss, global_step=self.global_step)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

class CompatibilityGAE(Model):
    def __init__(self, placeholders, n_out, input_dim, num_classes, num_support,
                 learning_rate, hidden, batch_norm=False,
                 multi=False, init='def',edge_c=True, **kwargs):
        super(CompatibilityGAE, self).__init__(**kwargs)
        self.edge_c=edge_c
        self.n_out = n_out
        self.inputs = placeholders['node_features']
        self.support = placeholders['support']
        self.dropout = placeholders['dropout']
        self.labels = placeholders['labels']

        if edge_c==True:
            self.attribute_labels = placeholders["attribute_labels"]
        self.r_indices = placeholders['row_indices']
        self.c_indices = placeholders['col_indices']
        self.is_train = placeholders['is_train']
        
        self.hidden = hidden
        self.num_classes = num_classes
        self.num_support = num_support
        self.input_dim = input_dim#2048
        self.learning_rate = learning_rate
        self.batch_norm = batch_norm
        self.init = init

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1.e-8)

        self.build()

    def _loss(self):
        """
        For mlp decoder.
        """
        if self.edge_c:
            print(self.output2.shape,self.attribute_labels.shape)
            self.attribute_labels,_ = tf.split(self.attribute_labels,[num_sup,int(self.attribute_labels.shape[-1]-num_sup)],1)            
            
            self.loss += -tf.reduce_mean( (self.labels*tf.log(tf.clip_by_value(self.outputs,1e-10,1))+\
                                           (1-self.labels)*tf.log(tf.clip_by_value(1-self.outputs,1e-10,1))) + \
                                         (lamda*tf.reduce_mean(self.attribute_labels*tf.log(tf.clip_by_value(self.output2,1e-10,1))+\
                                        (1-self.attribute_labels)*tf.log(tf.clip_by_value(1-self.output2,1e-10,1.0)),axis=1)) ) 
            
        else:
            print("Single Node loss")
            self.loss += tf.reduce_mean(-(self.labels*tf.log(tf.clip_by_value(self.outputs,1e-10,1))+(1-self.labels)*tf.log(tf.clip_by_value(1-self.outputs,1e-10,1))))
#            self.loss += sigmoid_cross_entropy(self.outputs, self.labels)
        
        tf.summary.scalar('loss', self.loss)

    def _confmat(self):
        self.confmat += softmax_confusion_matrix(self.outputs, self.labels)

    def _accuracy(self):
#        self.accuracy = sigmoid_accuracy(self.outputs, self.labels)
        self.accuracy = binary_accuracy(self.outputs,self.labels)

    def predict(self):
        return tf.cast(self.outputs >= 0.5, tf.int64)

    def _build(self):
        input_dim = self.input_dim
        act_funct = tf.nn.relu
        # stack of GCN layers as the encoder
        for l in range(len(self.hidden)):#self.hidden: [350,350,350] 3 hidden layers
            self.layers.append(GCN(input_dim=input_dim,
                                     output_dim=self.hidden[l],#output_dim:350
                                     support=self.support,
                                     num_support=self.num_support,
                                     act=act_funct,
                                     bias=not self.batch_norm,
                                     dropout=self.dropout,
                                     logging=self.logging,
                                     batch_norm=self.batch_norm,
                                     is_train=self.is_train,
                                     init=self.init))
            input_dim = self.hidden[l]

        input_dim = self.hidden[-1]#input_dim becomes 350

        # this is the decoder
        self.layers.append(MLPDecoder(num_classes=self.num_classes,
                                           r_indices=self.r_indices,
                                           c_indices=self.c_indices,
                                           input_dim=input_dim,#input_dim: 350
                                           dropout=0.,
                                           act=lambda x: x,
                                           logging=self.logging,
                                           n_out=self.n_out,
                                           use_bias=True))
