import tensorflow as tf
from tensorflow.python.ops.rnn_cell import *
from tensorflow.python.ops.rnn_cell_impl import  _Linear
from tensorflow import keras
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from keras import backend as K

class Time4AILSTMCell(RNNCell):

  def __init__(self, num_units,
               use_peepholes=False, cell_clip=None,
               initializer=None, num_proj=None, proj_clip=None,
               num_unit_shards=None, num_proj_shards=None,
               forget_bias=1.0, state_is_tuple=True,
               activation=None, reuse=None):

    super(Time4AILSTMCell, self).__init__(_reuse=reuse)
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
    if num_unit_shards is not None or num_proj_shards is not None:
      logging.warn(
          "%s: The num_unit_shards and proj_unit_shards parameters are "
          "deprecated and will be removed in Jan 2017.  "
          "Use a variable scope with a partitioner instead.", self)

    self._num_units = num_units
    self._use_peepholes = use_peepholes
    self._cell_clip = cell_clip
    self._initializer = initializer
    self._num_proj = num_proj
    self._proj_clip = proj_clip
    self._num_unit_shards = num_unit_shards
    self._num_proj_shards = num_proj_shards
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation or math_ops.tanh

    if num_proj:
      self._state_size = (
          LSTMStateTuple(num_units, num_proj)
          if state_is_tuple else num_units + num_proj)
      self._output_size = num_proj
    else:
      self._state_size = (
          LSTMStateTuple(num_units, num_units)
          if state_is_tuple else 2 * num_units)
      self._output_size = num_units
    self._linear1 = None
    self._linear2 = None
    self._time_input_w1 = None
    self._time_input_w2 = None
    self._time_kernel_w1 = None
    self._time_kernel_t1 = None
    self._time_bias1 = None
    self._time_kernel_w2 = None
    self._time_kernel_t2 = None
    self._time_bias2 = None
    self._o_kernel_t1 = None
    self._o_kernel_t2 = None
    if self._use_peepholes:
      self._w_f_diag = None
      self._w_i_diag = None
      self._w_o_diag = None

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size
    
  def __call__(self, inputs, state, att_score):
      return self.call(inputs, state, att_score)
      
  def call(self, inputs, state, att_score=None):
    time_now_score = tf.expand_dims(inputs[:,-1], -1)
    time_last_score = tf.expand_dims(inputs[:,-2], -1)
    inputs = inputs[:,:-2]
    inputs = inputs * att_score
    num_proj = self._num_units if self._num_proj is None else self._num_proj
    sigmoid = math_ops.sigmoid

    if self._state_is_tuple:
      (c_prev, m_prev) = state
    else:
      c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
      m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])

    dtype = inputs.dtype
    input_size = inputs.get_shape().with_rank(2)[1]
    if input_size.value is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
            
    if self._time_kernel_w1 is None:
      scope = vs.get_variable_scope()
      with vs.variable_scope(
          scope, initializer=self._initializer) as unit_scope:
        with vs.variable_scope(unit_scope):
          self._time_input_w1 = vs.get_variable(
              "_time_input_w1", shape=[self._num_units], dtype=dtype)
          self._time_input_bias1 = vs.get_variable(
              "_time_input_bias1", shape=[self._num_units], dtype=dtype)
          self._time_input_w2 = vs.get_variable(
              "_time_input_w2", shape=[self._num_units], dtype=dtype)
          self._time_input_bias2 = vs.get_variable(
              "_time_input_bias2", shape=[self._num_units], dtype=dtype)
          self._time_kernel_w1 = vs.get_variable(
              "_time_kernel_w1", shape=[input_size, self._num_units], dtype=dtype)
          self._time_kernel_t1 = vs.get_variable(
              "_time_kernel_t1", shape=[self._num_units, self._num_units], dtype=dtype)
          self._time_bias1 = vs.get_variable(
              "_time_bias1", shape=[self._num_units], dtype=dtype)
          self._time_kernel_w2 = vs.get_variable(
              "_time_kernel_w2", shape=[input_size, self._num_units], dtype=dtype)
          self._time_kernel_t2 = vs.get_variable(
              "_time_kernel_t2", shape=[self._num_units, self._num_units], dtype=dtype)
          self._time_bias2 = vs.get_variable(
              "_time_bias2", shape=[self._num_units], dtype=dtype)
          self._o_kernel_t1 = vs.get_variable(
              "_o_kernel_t1", shape=[self._num_units, self._num_units], dtype=dtype)    
          self._o_kernel_t2 = vs.get_variable(
              "_o_kernel_t2", shape=[self._num_units, self._num_units], dtype=dtype)  
                
    time_now_input = tf.nn.tanh(time_now_score * self._time_input_w1 + self._time_input_bias1)
    time_last_input = tf.nn.tanh(time_last_score * self._time_input_w2 + self._time_input_bias2)      

    time_now_state = math_ops.matmul(inputs, self._time_kernel_w1) + math_ops.matmul(time_now_input, self._time_kernel_t1) + self._time_bias1
    time_last_state = math_ops.matmul(inputs, self._time_kernel_w2) + math_ops.matmul(time_last_input, self._time_kernel_t2) + self._time_bias2
    
    if self._linear1 is None:
      scope = vs.get_variable_scope()
      with vs.variable_scope(
          scope, initializer=self._initializer) as unit_scope:
        if self._num_unit_shards is not None:
          unit_scope.set_partitioner(
              partitioned_variables.fixed_size_partitioner(
                  self._num_unit_shards))
        self._linear1 = _Linear([inputs, m_prev], 4 * self._num_units, True)

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    lstm_matrix = self._linear1([inputs, m_prev])
    i, j, f, o = array_ops.split(
        value=lstm_matrix, num_or_size_splits=4, axis=1)
    o = o + math_ops.matmul(time_now_input, self._o_kernel_t1) + math_ops.matmul(time_last_input, self._o_kernel_t2)   
    # Diagonal connections
    if self._use_peepholes and not self._w_f_diag:
      scope = vs.get_variable_scope()
      with vs.variable_scope(
          scope, initializer=self._initializer) as unit_scope:
        with vs.variable_scope(unit_scope):
          self._w_f_diag = vs.get_variable(
              "w_f_diag", shape=[self._num_units], dtype=dtype)
          self._w_i_diag = vs.get_variable(
              "w_i_diag", shape=[self._num_units], dtype=dtype)
          self._w_o_diag = vs.get_variable(
              "w_o_diag", shape=[self._num_units], dtype=dtype)

    if self._use_peepholes:
      c = (sigmoid(f + self._forget_bias + self._w_f_diag * c_prev) * sigmoid(time_last_state) * c_prev +
           sigmoid(i + self._w_i_diag * c_prev) * sigmoid(time_now_state) * self._activation(j))
    else:
      c = (sigmoid(f + self._forget_bias) * sigmoid(time_last_state) * c_prev + sigmoid(i) * sigmoid(time_now_state) * self._activation(j))

    if self._cell_clip is not None:
      # pylint: disable=invalid-unary-operand-type
      c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
      # pylint: enable=invalid-unary-operand-type
    if self._use_peepholes:
      m = sigmoid(o + self._w_o_diag * c) * self._activation(c)
    else:
      m = sigmoid(o) * self._activation(c)

    if self._num_proj is not None:
      if self._linear2 is None:
        scope = vs.get_variable_scope()
        with vs.variable_scope(scope, initializer=self._initializer):
          with vs.variable_scope("projection") as proj_scope:
            if self._num_proj_shards is not None:
              proj_scope.set_partitioner(
                  partitioned_variables.fixed_size_partitioner(
                      self._num_proj_shards))
            self._linear2 = _Linear(m, self._num_proj, False)
      m = self._linear2(m)

      if self._proj_clip is not None:
        # pylint: disable=invalid-unary-operand-type
        m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)
        # pylint: enable=invalid-unary-operand-type

    new_state = (LSTMStateTuple(c, m) if self._state_is_tuple else
                 array_ops.concat([c, m], 1))
    return m, new_state

class Time4ALSTMCell(RNNCell):

  def __init__(self, num_units,
               use_peepholes=False, cell_clip=None,
               initializer=None, num_proj=None, proj_clip=None,
               num_unit_shards=None, num_proj_shards=None,
               forget_bias=1.0, state_is_tuple=True,
               activation=None, reuse=None):

    super(Time4ALSTMCell, self).__init__(_reuse=reuse)
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
    if num_unit_shards is not None or num_proj_shards is not None:
      logging.warn(
          "%s: The num_unit_shards and proj_unit_shards parameters are "
          "deprecated and will be removed in Jan 2017.  "
          "Use a variable scope with a partitioner instead.", self)

    self._num_units = num_units
    self._use_peepholes = use_peepholes
    self._cell_clip = cell_clip
    self._initializer = initializer
    self._num_proj = num_proj
    self._proj_clip = proj_clip
    self._num_unit_shards = num_unit_shards
    self._num_proj_shards = num_proj_shards
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation or math_ops.tanh

    if num_proj:
      self._state_size = (
          LSTMStateTuple(num_units, num_proj)
          if state_is_tuple else num_units + num_proj)
      self._output_size = num_proj
    else:
      self._state_size = (
          LSTMStateTuple(num_units, num_units)
          if state_is_tuple else 2 * num_units)
      self._output_size = num_units
    self._linear1 = None
    self._linear2 = None
    self._time_input_w1 = None
    self._time_input_w2 = None
    self._time_kernel_w1 = None
    self._time_kernel_t1 = None
    self._time_bias1 = None
    self._time_kernel_w2 = None
    self._time_kernel_t2 = None
    self._time_bias2 = None
    self._o_kernel_t1 = None
    self._o_kernel_t2 = None
    if self._use_peepholes:
      self._w_f_diag = None
      self._w_i_diag = None
      self._w_o_diag = None

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size
    
  def __call__(self, inputs, state, att_score):
      return self.call(inputs, state, att_score)
      
  def call(self, inputs, state, att_score=None):
    time_now_score = tf.expand_dims(inputs[:,-1], -1)
    time_last_score = tf.expand_dims(inputs[:,-2], -1)
    inputs = inputs[:,:-2]
    num_proj = self._num_units if self._num_proj is None else self._num_proj
    sigmoid = math_ops.sigmoid

    if self._state_is_tuple:
      (c_prev, m_prev) = state
    else:
      c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
      m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])

    dtype = inputs.dtype
    input_size = inputs.get_shape().with_rank(2)[1]
    if input_size.value is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
            
    if self._time_kernel_w1 is None:
      scope = vs.get_variable_scope()
      with vs.variable_scope(
          scope, initializer=self._initializer) as unit_scope:
        with vs.variable_scope(unit_scope):
          self._time_input_w1 = vs.get_variable(
              "_time_input_w1", shape=[self._num_units], dtype=dtype)
          self._time_input_bias1 = vs.get_variable(
              "_time_input_bias1", shape=[self._num_units], dtype=dtype)
          self._time_input_w2 = vs.get_variable(
              "_time_input_w2", shape=[self._num_units], dtype=dtype)
          self._time_input_bias2 = vs.get_variable(
              "_time_input_bias2", shape=[self._num_units], dtype=dtype)
          self._time_kernel_w1 = vs.get_variable(
              "_time_kernel_w1", shape=[input_size, self._num_units], dtype=dtype)
          self._time_kernel_t1 = vs.get_variable(
              "_time_kernel_t1", shape=[self._num_units, self._num_units], dtype=dtype)
          self._time_bias1 = vs.get_variable(
              "_time_bias1", shape=[self._num_units], dtype=dtype)
          self._time_kernel_w2 = vs.get_variable(
              "_time_kernel_w2", shape=[input_size, self._num_units], dtype=dtype)
          self._time_kernel_t2 = vs.get_variable(
              "_time_kernel_t2", shape=[self._num_units, self._num_units], dtype=dtype)
          self._time_bias2 = vs.get_variable(
              "_time_bias2", shape=[self._num_units], dtype=dtype)
          self._o_kernel_t1 = vs.get_variable(
              "_o_kernel_t1", shape=[self._num_units, self._num_units], dtype=dtype)    
          self._o_kernel_t2 = vs.get_variable(
              "_o_kernel_t2", shape=[self._num_units, self._num_units], dtype=dtype)  
                
    time_now_input = tf.nn.tanh(time_now_score * self._time_input_w1 + self._time_input_bias1)
    time_last_input = tf.nn.tanh(time_last_score * self._time_input_w2 + self._time_input_bias2)      

    time_now_state = math_ops.matmul(inputs, self._time_kernel_w1) + math_ops.matmul(time_now_input, self._time_kernel_t1) + self._time_bias1
    time_last_state = math_ops.matmul(inputs, self._time_kernel_w2) + math_ops.matmul(time_last_input, self._time_kernel_t2) + self._time_bias2
    
    if self._linear1 is None:
      scope = vs.get_variable_scope()
      with vs.variable_scope(
          scope, initializer=self._initializer) as unit_scope:
        if self._num_unit_shards is not None:
          unit_scope.set_partitioner(
              partitioned_variables.fixed_size_partitioner(
                  self._num_unit_shards))
        self._linear1 = _Linear([inputs, m_prev], 4 * self._num_units, True)

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    lstm_matrix = self._linear1([inputs, m_prev])
    i, j, f, o = array_ops.split(
        value=lstm_matrix, num_or_size_splits=4, axis=1)
    o = o + math_ops.matmul(time_now_input, self._o_kernel_t1) + math_ops.matmul(time_last_input, self._o_kernel_t2)   
    # Diagonal connections
    if self._use_peepholes and not self._w_f_diag:
      scope = vs.get_variable_scope()
      with vs.variable_scope(
          scope, initializer=self._initializer) as unit_scope:
        with vs.variable_scope(unit_scope):
          self._w_f_diag = vs.get_variable(
              "w_f_diag", shape=[self._num_units], dtype=dtype)
          self._w_i_diag = vs.get_variable(
              "w_i_diag", shape=[self._num_units], dtype=dtype)
          self._w_o_diag = vs.get_variable(
              "w_o_diag", shape=[self._num_units], dtype=dtype)

    if self._use_peepholes:
      c = (sigmoid(f + self._forget_bias + self._w_f_diag * c_prev) * sigmoid(time_last_state) * c_prev +
           sigmoid(i + self._w_i_diag * c_prev) * sigmoid(time_now_state) * self._activation(j))
    else:
      c = (sigmoid(f + self._forget_bias) * sigmoid(time_last_state) * c_prev + sigmoid(i) * sigmoid(time_now_state) * self._activation(j))

    if self._cell_clip is not None:
      # pylint: disable=invalid-unary-operand-type
      c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
      # pylint: enable=invalid-unary-operand-type
    if self._use_peepholes:
      m = sigmoid(o + self._w_o_diag * c) * self._activation(c)
    else:
      m = sigmoid(o) * self._activation(c)

    if self._num_proj is not None:
      if self._linear2 is None:
        scope = vs.get_variable_scope()
        with vs.variable_scope(scope, initializer=self._initializer):
          with vs.variable_scope("projection") as proj_scope:
            if self._num_proj_shards is not None:
              proj_scope.set_partitioner(
                  partitioned_variables.fixed_size_partitioner(
                      self._num_proj_shards))
            self._linear2 = _Linear(m, self._num_proj, False)
      m = self._linear2(m)

      if self._proj_clip is not None:
        # pylint: disable=invalid-unary-operand-type
        m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)
        # pylint: enable=invalid-unary-operand-type
    c = att_score * c + (1. - att_score) * c_prev
    m = att_score * m + (1. - att_score) * m_prev
    new_state = (LSTMStateTuple(c, m) if self._state_is_tuple else
                 array_ops.concat([c, m], 1))
    return m, new_state
    
class VecAttGRUCell(RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
  Args:
    num_units: int, The number of units in the GRU cell.
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    kernel_initializer: (optional) The initializer to use for the weight and
    projection matrices.
    bias_initializer: (optional) The initializer to use for the bias.
  """

  def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None):
    super(VecAttGRUCell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._activation = activation or math_ops.tanh
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._gate_linear = None
    self._candidate_linear = None

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units
  def __call__(self, inputs, state, att_score):
      return self.call(inputs, state, att_score)
  def call(self, inputs, state, att_score=None):
    """Gated recurrent unit (GRU) with nunits cells."""
    if self._gate_linear is None:
      bias_ones = self._bias_initializer
      if self._bias_initializer is None:
        bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
      with vs.variable_scope("gates"):  # Reset gate and update gate.
        self._gate_linear = _Linear(
            [inputs, state],
            2 * self._num_units,
            True,
            bias_initializer=bias_ones,
            kernel_initializer=self._kernel_initializer)

    value = math_ops.sigmoid(self._gate_linear([inputs, state]))
    r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

    r_state = r * state
    if self._candidate_linear is None:
      with vs.variable_scope("candidate"):
        self._candidate_linear = _Linear(
            [inputs, r_state],
            self._num_units,
            True,
            bias_initializer=self._bias_initializer,
            kernel_initializer=self._kernel_initializer)
    c = self._activation(self._candidate_linear([inputs, r_state]))
    u = (1.0 - att_score) * u
    new_h = u * state + (1 - u) * c
    return new_h, new_h
    
def prelu(_x, scope=''):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu_"+scope, shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)

def calc_auc(raw_arr):
    """Summary

    Args:
        raw_arr (TYPE): Description

    Returns:
        TYPE: Description
    """

    arr = sorted(raw_arr, key=lambda d:d[0], reverse=True)
    pos, neg = 0., 0.
    for record in arr:
        if record[1] == 1.:
            pos += 1
        else:
            neg += 1

    fp, tp = 0., 0.
    xy_arr = []
    for record in arr:
        if record[1] == 1.:
            tp += 1
        else:
            fp += 1
        xy_arr.append([fp/neg, tp/pos])

    auc = 0.
    prev_x = 0.
    prev_y = 0.
    for x, y in xy_arr:
        if x != prev_x:
            auc += ((x - prev_x) * (y + prev_y) / 2.)
            prev_x = x
            prev_y = y

    return auc

def attention(query, facts, attention_size, mask, stag='null', mode='LIST', softmax_stag=1, time_major=False, return_alphas=False):
    if isinstance(facts, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        facts = tf.concat(facts, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])

    mask = tf.equal(mask, tf.ones_like(mask))
    hidden_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
    input_size = query.get_shape().as_list()[-1]

    # Trainable parameters
    w1 = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    w2 = tf.Variable(tf.random_normal([input_size, attention_size], stddev=0.1))
    b = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    v = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `tmp` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        tmp1 = tf.tensordot(facts, w1, axes=1)
        tmp2 = tf.tensordot(query, w2, axes=1)
        tmp2 = tf.reshape(tmp2, [-1, 1, tf.shape(tmp2)[-1]])
        tmp = tf.tanh((tmp1 + tmp2) + b)

    # For each of the timestamps its vector of size A from `tmp` is reduced with `v` vector
    v_dot_tmp = tf.tensordot(tmp, v, axes=1, name='v_dot_tmp')  # (B,T) shape
    key_masks = mask # [B, 1, T]
    # key_masks = tf.expand_dims(mask, 1) # [B, 1, T]
    paddings = tf.ones_like(v_dot_tmp) * (-2 ** 32 + 1)
    v_dot_tmp = tf.where(key_masks, v_dot_tmp, paddings)  # [B, 1, T]
    alphas = tf.nn.softmax(v_dot_tmp, name='alphas')         # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    #output = tf.reduce_sum(facts * tf.expand_dims(alphas, -1), 1)
    output = facts * tf.expand_dims(alphas, -1)
    output = tf.reshape(output, tf.shape(facts))
    # output = output / (facts.get_shape().as_list()[-1] ** 0.5)
    if not return_alphas:
        return output
    else:
        return output, alphas

def attention_query(query, facts, attention_size, time_major=False, return_alphas=False):

    if isinstance(facts, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        facts = tf.concat(facts, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])

    hidden_size_facts = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
    hidden_size_query = query.get_shape().as_list()[-1]
    # Trainable parameters
    w1_omega = tf.Variable(tf.random_normal([hidden_size_facts, attention_size], stddev=0.1))
    w2_omega = tf.Variable(tf.random_normal([hidden_size_query, attention_size], stddev=0.1))
    b1_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    b2_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        v1 = tf.tanh(tf.tensordot(facts, w1_omega, axes=1) + b1_omega)
        v2 = tf.tanh(tf.tensordot(query, w2_omega, axes=1) + b2_omega)

    v2 = tf.reshape(tf.tile(v2,[1,tf.shape(facts)[1]]),[-1,tf.shape(facts)[1],attention_size])
    qv = tf.multiply(v1, v2)
    vu = tf.reduce_sum(qv, axis=-1)  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape
    alphas = tf.reshape(alphas, [-1, tf.shape(facts)[1]])
    output = facts * tf.expand_dims(alphas, -1)
    output = tf.reshape(output, tf.shape(facts))
    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    #output = tf.reduce_sum(facts * tf.expand_dims(alphas, -1), 1)
    #output = tf.matmul(alphas, facts)
    if not return_alphas:
        return output
    else:
        return output, alphas

def attention_HAN(inputs, attention_size=None, time_major=False, return_alphas=False):

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])
        
    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer    
    if attention_size == None:
        attention_size = hidden_size
    
    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape
    alphas = tf.reshape(alphas, [-1, tf.shape(inputs)[1]])
    output = inputs * tf.expand_dims(alphas, -1)
    output = tf.reshape(output, tf.shape(inputs))    
    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    #output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas
                
def din_attention(query, facts, attention_size, mask, stag='null', mode='SUM', softmax_stag=1, time_major=False, return_alphas=False):
    if isinstance(facts, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        facts = tf.concat(facts, 2)
        print ("querry_size mismatch")
        query = tf.concat(values = [
        query,
        query,
        ], axis=1)

    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])
    mask = tf.equal(mask, tf.ones_like(mask))
    facts_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
    querry_size = query.get_shape().as_list()[-1]
    queries = tf.tile(query, [1, tf.shape(facts)[1]])
    queries = tf.reshape(queries, tf.shape(facts))
    din_all = tf.concat([queries, facts, queries-facts, queries*facts], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])
    scores = d_layer_3_all
    # Mask
    # key_masks = tf.sequence_mask(facts_length, tf.shape(facts)[1])   # [B, T]
    key_masks = tf.expand_dims(mask, 1) # [B, 1, T]
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]

    # Scale
    # scores = scores / (facts.get_shape().as_list()[-1] ** 0.5)

    # Activation
    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [B, 1, T]

    # Weighted sum
    if mode == 'SUM':
        output = tf.matmul(scores, facts)  # [B, 1, H]
        # output = tf.reshape(output, [-1, tf.shape(facts)[-1]])
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
        output = facts * tf.expand_dims(scores, -1)
        output = tf.reshape(output, tf.shape(facts))
    return output

def din_fcn_attention(query, facts, attention_size, mask, stag='null', mode='SUM', softmax_stag=1, time_major=False, return_alphas=False, forCnn=False, scope = ""):
    if isinstance(facts, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        facts = tf.concat(facts, 2)
    if len(facts.get_shape().as_list()) == 2:
        facts = tf.expand_dims(facts, 1)

    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])
    # Trainable parameters
    mask = tf.equal(mask, tf.ones_like(mask))
    facts_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
    querry_size = query.get_shape().as_list()[-1]
    query = tf.layers.dense(query, facts_size, activation=None, name='f1' + stag)
    query = prelu(query,scope=scope)
    queries = tf.tile(query, [1, tf.shape(facts)[1]])
    queries = tf.reshape(queries, tf.shape(facts))
    din_all = tf.concat([queries, facts, queries-facts, queries*facts], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])
    scores = d_layer_3_all
    # Mask
    # key_masks = tf.sequence_mask(facts_length, tf.shape(facts)[1])   # [B, T]
    key_masks = tf.expand_dims(mask, 1) # [B, 1, T]
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    if not forCnn:
        scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]

    # Scale
    # scores = scores / (facts.get_shape().as_list()[-1] ** 0.5)

    # Activation
    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [B, 1, T]

    # Weighted sum
    if mode == 'SUM':
        output = tf.matmul(scores, facts)  # [B, 1, H]
        # output = tf.reshape(output, [-1, tf.shape(facts)[-1]])
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
        output = facts * tf.expand_dims(scores, -1)
        output = tf.reshape(output, tf.shape(facts))
    if return_alphas:
        return output, scores
    return output

def self_attention(facts, ATTENTION_SIZE, mask, stag='null'):
    if len(facts.get_shape().as_list()) == 2:
        facts = tf.expand_dims(facts, 1)

    def cond(batch, output, i):
        return tf.less(i, tf.shape(batch)[1])

    def body(batch, output, i):
        self_attention_tmp = din_fcn_attention(batch[:, i, :], batch[:, 0:i+1, :],
                                               ATTENTION_SIZE, mask[:, 0:i+1], softmax_stag=1, stag=stag,
                                               mode='LIST')
        self_attention_tmp = tf.reduce_sum(self_attention_tmp, 1)
        output = output.write(i, self_attention_tmp)
        return batch, output, i + 1

    output_ta = tf.TensorArray(dtype=tf.float32,
                               size=0,
                               dynamic_size=True,
                               element_shape=(facts[:, 0, :].get_shape()))
    _, output_op, _ = tf.while_loop(cond, body, [facts, output_ta, 0])
    self_attention = output_op.stack()
    self_attention = tf.transpose(self_attention, perm = [1, 0, 2])
    return self_attention

def self_all_attention(facts, ATTENTION_SIZE, mask, stag='null'):
    if len(facts.get_shape().as_list()) == 2:
        facts = tf.expand_dims(facts, 1)

    def cond(batch, output, i):
        return tf.less(i, tf.shape(batch)[1])

    def body(batch, output, i):
        self_attention_tmp = din_fcn_attention(batch[:, i, :], batch,
                                               ATTENTION_SIZE, mask, softmax_stag=1, stag=stag,
                                               mode='LIST')
        self_attention_tmp = tf.reduce_sum(self_attention_tmp, 1)
        output = output.write(i, self_attention_tmp)
        return batch, output, i + 1

    output_ta = tf.TensorArray(dtype=tf.float32,
                               size=0,
                               dynamic_size=True,
                               element_shape=(facts[:, 0, :].get_shape()))
    _, output_op, _ = tf.while_loop(cond, body, [facts, output_ta, 0])
    self_attention = output_op.stack()
    self_attention = tf.transpose(self_attention, perm = [1, 0, 2])
    return self_attention

def din_fcn_shine(query, facts, attention_size, mask, stag='null', mode='SUM', softmax_stag=1, time_major=False, return_alphas=False):
    if isinstance(facts, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        facts = tf.concat(facts, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])
    # Trainable parameters
    mask = tf.equal(mask, tf.ones_like(mask))
    facts_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
    querry_size = query.get_shape().as_list()[-1]
    query = tf.layers.dense(query, facts_size, activation=None, name='f1_trans_shine' + stag)
    query = prelu(query)
    queries = tf.tile(query, [1, tf.shape(facts)[1]])
    queries = tf.reshape(queries, tf.shape(facts))
    din_all = tf.concat([queries, facts, queries-facts, queries*facts], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, facts_size, activation=tf.nn.sigmoid, name='f1_shine_att' + stag)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, facts_size, activation=tf.nn.sigmoid, name='f2_shine_att' + stag)
    d_layer_2_all = tf.reshape(d_layer_2_all, tf.shape(facts))
    output = d_layer_2_all
    return output

