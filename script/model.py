import tensorflow as tf
from rnn_cell_impl import GRUCell, LSTMCell, Time1LSTMCell, Time2LSTMCell, Time3LSTMCell, Time4LSTMCell, CARNNCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from rnn import dynamic_rnn
from utils import *
from Dice import dice

class Model(object):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling = False):
        with tf.name_scope('Inputs'):
            self.mid_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='mid_his_batch_ph')
            self.cat_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='cat_his_batch_ph')
            self.time_his_batch_ph = tf.placeholder(tf.float32, [None, None], name='time_his_batch_ph')
            self.time_last_his_batch_ph = tf.placeholder(tf.float32, [None, None], name='time_last_his_batch_ph')
            self.time_now_his_batch_ph = tf.placeholder(tf.float32, [None, None], name='time_now_his_batch_ph')
            self.uid_batch_ph = tf.placeholder(tf.int32, [None, ], name='uid_batch_ph')
            self.mid_batch_ph = tf.placeholder(tf.int32, [None, ], name='mid_batch_ph')
            self.cat_batch_ph = tf.placeholder(tf.int32, [None, ], name='cat_batch_ph')
            self.mask = tf.placeholder(tf.float32, [None, None], name='mask')
            self.seq_len_ph = tf.placeholder(tf.int32, [None], name='seq_len_ph')
            self.target_ph = tf.placeholder(tf.float32, [None, None], name='target_ph')
            self.lr = tf.placeholder(tf.float64, [])
            self.use_negsampling =use_negsampling
            if use_negsampling:
                self.noclk_mid_batch_ph = tf.placeholder(tf.int32, [None, None, None], name='noclk_mid_batch_ph')
                self.noclk_cat_batch_ph = tf.placeholder(tf.int32, [None, None, None], name='noclk_cat_batch_ph')

        # Embedding layer
        with tf.name_scope('Embedding_layer'):
            self.uid_embeddings_var = tf.get_variable("uid_embedding_var", [n_uid, EMBEDDING_DIM])
            tf.summary.histogram('uid_embeddings_var', self.uid_embeddings_var)
            self.uid_batch_embedded = tf.nn.embedding_lookup(self.uid_embeddings_var, self.uid_batch_ph)

            self.mid_embeddings_var = tf.get_variable("mid_embedding_var", [n_mid, EMBEDDING_DIM])
            tf.summary.histogram('mid_embeddings_var', self.mid_embeddings_var)
            self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_batch_ph)
            self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_his_batch_ph)
            if self.use_negsampling:
                self.noclk_mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.noclk_mid_batch_ph)

            self.cat_embeddings_var = tf.get_variable("cat_embedding_var", [n_cat, EMBEDDING_DIM])
            tf.summary.histogram('cat_embeddings_var', self.cat_embeddings_var)
            self.cat_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_batch_ph)
            self.cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_his_batch_ph)
            if self.use_negsampling:
                self.noclk_cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.noclk_cat_batch_ph)

        self.item_eb = tf.concat([self.mid_batch_embedded, self.cat_batch_embedded], 1)
        self.item_his_eb = tf.concat([self.mid_his_batch_embedded, self.cat_his_batch_embedded], 2)
        
        if self.use_negsampling:
            self.noclk_item_his_eb = tf.concat(
                [self.noclk_mid_his_batch_embedded[:, :, 0, :], self.noclk_cat_his_batch_embedded[:, :, 0, :]], -1)
            self.noclk_item_his_eb = tf.reshape(self.noclk_item_his_eb,
                                                [-1, tf.shape(self.noclk_mid_his_batch_embedded)[1], 36])

            self.noclk_his_eb = tf.concat([self.noclk_mid_his_batch_embedded, self.noclk_cat_his_batch_embedded], -1)
            self.noclk_his_eb_sum_1 = tf.reduce_sum(self.noclk_his_eb, 2)
            self.noclk_his_eb_sum = tf.reduce_sum(self.noclk_his_eb_sum_1, 1)
            
    def build_fcn_net(self, inp, use_dice = False):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        dnn1 = tf.layers.dense(bn1, 200, activation=None, name='f1')
        if use_dice:
            dnn1 = dice(dnn1, name='dice_1')
        else:
            dnn1 = prelu(dnn1, 'prelu1')

        dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='f2')
        if use_dice:
            dnn2 = dice(dnn2, name='dice_2')
        else:
            dnn2 = prelu(dnn2, 'prelu2')
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')
        self.y_hat = tf.nn.softmax(dnn3) + 0.00000001

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
            self.loss = ctr_loss
            if self.use_negsampling:
                self.loss += self.aux_loss
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()

    def auxiliary_loss(self, h_states, click_seq, noclick_seq, mask, stag = None):
        mask = tf.cast(mask, tf.float32)
        click_input_ = tf.concat([h_states, click_seq], -1)
        noclick_input_ = tf.concat([h_states, noclick_seq], -1)
        click_prop_ = self.auxiliary_net(click_input_, stag = stag)[:, :, 0]
        noclick_prop_ = self.auxiliary_net(noclick_input_, stag = stag)[:, :, 0]
        click_loss_ = - tf.reshape(tf.log(click_prop_), [-1, tf.shape(click_seq)[1]]) * mask
        noclick_loss_ = - tf.reshape(tf.log(1.0 - noclick_prop_), [-1, tf.shape(noclick_seq)[1]]) * mask
        loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)
        return loss_

    def auxiliary_net(self, in_, stag='auxiliary_net'):
        bn1 = tf.layers.batch_normalization(inputs=in_, name='bn1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.layers.dense(bn1, 100, activation=None, name='f1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.nn.sigmoid(dnn1)
        dnn2 = tf.layers.dense(dnn1, 50, activation=None, name='f2' + stag, reuse=tf.AUTO_REUSE)
        dnn2 = tf.nn.sigmoid(dnn2)
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3' + stag, reuse=tf.AUTO_REUSE)
        y_hat = tf.nn.softmax(dnn3) + 0.00000001
        return y_hat


    def train(self, sess, inps):
        if self.use_negsampling:
            loss, accuracy, aux_loss, _ = sess.run([self.loss, self.accuracy, self.aux_loss, self.optimizer], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cat_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cat_his_batch_ph: inps[4],
                self.time_his_batch_ph: inps[5],
                self.time_last_his_batch_ph: inps[6],
                self.time_now_his_batch_ph: inps[7],
                self.mask: inps[8],
                self.target_ph: inps[9],
                self.seq_len_ph: inps[10],
                self.lr: inps[11],
                self.noclk_mid_batch_ph: inps[12],
                self.noclk_cat_batch_ph: inps[13],
            })
            return loss, accuracy, aux_loss
        else:
            loss, accuracy, _ = sess.run([self.loss, self.accuracy, self.optimizer], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cat_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cat_his_batch_ph: inps[4],
                self.time_his_batch_ph: inps[5],
                self.time_last_his_batch_ph: inps[6],
                self.time_now_his_batch_ph: inps[7],
                self.mask: inps[8],
                self.target_ph: inps[9],
                self.seq_len_ph: inps[10],
                self.lr: inps[11],
            })
            return loss, accuracy, 0

    def calculate(self, sess, inps):
        if self.use_negsampling:
            probs, loss, accuracy, aux_loss = sess.run([self.y_hat, self.loss, self.accuracy, self.aux_loss], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cat_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cat_his_batch_ph: inps[4],
                self.time_his_batch_ph: inps[5],
                self.time_last_his_batch_ph: inps[6],
                self.time_now_his_batch_ph: inps[7],
                self.mask: inps[8],
                self.target_ph: inps[9],
                self.seq_len_ph: inps[10],
                self.noclk_mid_batch_ph: inps[11],
                self.noclk_cat_batch_ph: inps[12],
            })
            return probs, loss, accuracy, aux_loss
        else:
            probs, loss, accuracy = sess.run([self.y_hat, self.loss, self.accuracy], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cat_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cat_his_batch_ph: inps[4],
                self.time_his_batch_ph: inps[5],
                self.time_last_his_batch_ph: inps[6],
                self.time_now_his_batch_ph: inps[7],
                self.mask: inps[8],
                self.target_ph: inps[9],
                self.seq_len_ph: inps[10]
            })
            return probs, loss, accuracy, 0

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path)

class Model_ASVD(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_ASVD, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                                          ATTENTION_SIZE,
                                                          use_negsampling)
                                                          
        self.item_his_eb_sum = tf.reduce_sum(self.item_his_eb, 1)
        inp = tf.concat([self.item_eb, self.item_his_eb_sum], 1)
        self.build_fcn_net(inp, use_dice=False)
                
class Model_DIN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_DIN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                           ATTENTION_SIZE,
                                           use_negsampling)

        # Attention layer
        with tf.name_scope('Attention_layer'):
            attention_output = din_attention(self.item_eb, self.item_his_eb, ATTENTION_SIZE, self.mask)
            att_fea = tf.reduce_sum(attention_output, 1)
            tf.summary.histogram('att_fea', att_fea)
            
        inp = tf.concat([self.item_eb, att_fea], -1)
        self.build_fcn_net(inp, use_dice=True)

class Model_LSTM(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_LSTM, self).__init__(n_uid, n_mid, n_cat,
                                                       EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                       use_negsampling)

        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, final_state1 = dynamic_rnn(LSTMCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="lstm1")
            tf.summary.histogram('LSTM_outputs', rnn_outputs)

        inp = tf.concat([self.item_eb, final_state1[1]], 1)
        self.build_fcn_net(inp, use_dice=True)

class Model_LSTMPP(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_LSTMPP, self).__init__(n_uid, n_mid, n_cat,
                                                       EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                       use_negsampling)
        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_outputs1, alphas1 = attention_HAN(self.item_his_eb, attention_size=ATTENTION_SIZE, return_alphas=True)
            att_fea1 = tf.reduce_sum(att_outputs1, 1)
            tf.summary.histogram('att_fea1', att_fea1)
            
        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, final_state1 = dynamic_rnn(LSTMCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="lstm1")
            tf.summary.histogram('LSTM_outputs', rnn_outputs)
         
        #alpha            
        with tf.name_scope('User_alpha'):    
            concat_all = tf.concat([self.item_eb, att_fea1, final_state1[1], tf.expand_dims(self.time_now_his_batch_ph[:,-1], -1)], 1)
            concat_att1 = tf.layers.dense(concat_all, 80, activation=tf.nn.sigmoid, name='concat_att1')
            concat_att2 = tf.layers.dense(concat_att1, 40, activation=tf.nn.sigmoid, name='concat_att2')
            user_alpha = tf.layers.dense(concat_att2, 1, activation=tf.nn.sigmoid, name='concat_att3') 
            user_embed = att_fea1 * user_alpha + final_state1[1] * (1.0 - user_alpha)

        inp = tf.concat([self.item_eb, user_embed], 1)
        self.build_fcn_net(inp, use_dice=True)
        
class Model_NARM(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_NARM, self).__init__(n_uid, n_mid, n_cat,
                                                       EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                       use_negsampling)
        # RNN layer(-s) - 1
        with tf.name_scope('rnn_1'):
            rnn_outputs1, final_state1 = dynamic_rnn(LSTMCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="lstm1")
            tf.summary.histogram('LSTM_outputs1', rnn_outputs1)

        # RNN layer(-s) - 2
        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(LSTMCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="lstm2")
            tf.summary.histogram('LSTM_outputs2', rnn_outputs2)
             
        # Attention layer
        with tf.name_scope('Attention_layer'):
            att_outputs, alphas = din_fcn_attention(final_state1[1], rnn_outputs2, ATTENTION_SIZE, self.mask, 
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
            tf.summary.histogram('alpha_outputs', alphas)
            att_fea = tf.reduce_sum(att_outputs, 1)

        inp = tf.concat([final_state1[1], att_fea, self.item_eb], 1)
        self.build_fcn_net(inp, use_dice=True)
        
class Model_CARNN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_CARNN, self).__init__(n_uid, n_mid, n_cat,
                                                       EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                       use_negsampling)
        self.item_his_eb = tf.concat([self.item_his_eb, tf.expand_dims(self.time_his_batch_ph, -1)], -1)
        # RNN layer(-s) - 1
        with tf.name_scope('rnn_1'):
            rnn_outputs1, final_state1 = dynamic_rnn(CARNNCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="carnn1")
            tf.summary.histogram('CARNN_outputs1', rnn_outputs1)

        inp = tf.concat([final_state1, self.item_eb], 1)
        self.build_fcn_net(inp, use_dice=True)
                                
class Model_Time1LSTM(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_Time1LSTM, self).__init__(n_uid, n_mid, n_cat,
                                                       EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                       use_negsampling)
                                                       
        self.item_his_eb = tf.concat([self.item_his_eb, tf.expand_dims(self.time_his_batch_ph, -1)], -1)
        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, final_state1 = dynamic_rnn(Time1LSTMCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="lstm1")
            tf.summary.histogram('LSTM_outputs', rnn_outputs)

        inp = tf.concat([self.item_eb, final_state1[1]], 1)
        self.build_fcn_net(inp, use_dice=True)
        
class Model_Time2LSTM(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_Time2LSTM, self).__init__(n_uid, n_mid, n_cat,
                                                       EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                       use_negsampling)
        
        self.item_his_eb = tf.concat([self.item_his_eb, tf.expand_dims(self.time_his_batch_ph, -1)], -1)
        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, final_state1 = dynamic_rnn(Time2LSTMCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="lstm1")
            tf.summary.histogram('LSTM_outputs', rnn_outputs)

        inp = tf.concat([self.item_eb, final_state1[1]], 1)
        self.build_fcn_net(inp, use_dice=True)
        
class Model_Time3LSTM(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_Time3LSTM, self).__init__(n_uid, n_mid, n_cat,
                                                       EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                       use_negsampling)
                                                       
        self.item_his_eb = tf.concat([self.item_his_eb, tf.expand_dims(self.time_his_batch_ph, -1)], -1)
        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, final_state1 = dynamic_rnn(Time3LSTMCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="lstm1")
            tf.summary.histogram('LSTM_outputs', rnn_outputs)

        inp = tf.concat([self.item_eb, final_state1[1]], 1)
        self.build_fcn_net(inp, use_dice=True)
        
class Model_DIEN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_DIEN, self).__init__(n_uid, n_mid, n_cat,
                                                          EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                          use_negsampling)

        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_outputs, alphas = din_fcn_attention(self.item_eb, rnn_outputs, ATTENTION_SIZE, self.mask,
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
            tf.summary.histogram('alpha_outputs', alphas)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs,
                                                     att_scores = tf.expand_dims(alphas, -1),
                                                     sequence_length=self.seq_len_ph, dtype=tf.float32,
                                                     scope="gru2")
            tf.summary.histogram('GRU2_Final_State', final_state2)

        inp = tf.concat([self.item_eb, final_state2], 1)
        self.build_fcn_net(inp, use_dice=True)

class Model_A2SVD(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_A2SVD, self).__init__(n_uid, n_mid, n_cat,
                                                       EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                       use_negsampling)
        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_outputs1, alphas1 = attention_HAN(self.item_his_eb, attention_size=ATTENTION_SIZE, return_alphas=True)
            att_fea1 = tf.reduce_sum(att_outputs1, 1)
            tf.summary.histogram('att_fea1', att_fea1)

        inp = tf.concat([self.item_eb, att_fea1], 1)
        self.build_fcn_net(inp, use_dice=True)
        
class Model_T_SeqRec(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_T_SeqRec, self).__init__(n_uid, n_mid, n_cat,
                                                       EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                       use_negsampling)

        item_his_eb_new = tf.concat([self.item_his_eb, tf.expand_dims(self.time_last_his_batch_ph, -1)], -1)
        item_his_eb_new = tf.concat([item_his_eb_new, tf.expand_dims(self.time_now_his_batch_ph, -1)], -1)
        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, final_state1 = dynamic_rnn(Time4LSTMCell(HIDDEN_SIZE), inputs=item_his_eb_new,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="lstm1")
            tf.summary.histogram('LSTM_outputs', rnn_outputs)

        inp = tf.concat([self.item_eb, final_state1[1]], 1)
        self.build_fcn_net(inp, use_dice=True)
        
class Model_TC_SeqRec_I(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_TC_SeqRec_I, self).__init__(n_uid, n_mid, n_cat,
                                                       EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                       use_negsampling)
        # Attention layer
        with tf.name_scope('Attention_layer'):
            att_outputs, alphas = din_fcn_attention(self.item_eb, self.item_his_eb, ATTENTION_SIZE, self.mask,
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
            tf.summary.histogram('alpha_outputs', alphas)
            
        item_his_eb_new = tf.concat([self.item_his_eb, tf.expand_dims(self.time_last_his_batch_ph, -1)], -1)
        item_his_eb_new = tf.concat([item_his_eb_new, tf.expand_dims(self.time_now_his_batch_ph, -1)], -1)
        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, final_state1 = dynamic_rnn(Time4AILSTMCell(HIDDEN_SIZE), inputs=item_his_eb_new,
                                         att_scores = tf.expand_dims(alphas, -1),
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="lstm1")
            tf.summary.histogram('LSTM_outputs', rnn_outputs)

        inp = tf.concat([self.item_eb, final_state1[1]], 1)
        self.build_fcn_net(inp, use_dice=True)
        
class Model_TC_SeqRec_G(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_TC_SeqRec_G, self).__init__(n_uid, n_mid, n_cat,
                                                       EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                       use_negsampling)
        # Attention layer
        with tf.name_scope('Attention_layer'):
            att_outputs, alphas = din_fcn_attention(self.item_eb, self.item_his_eb, ATTENTION_SIZE, self.mask,
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
            tf.summary.histogram('alpha_outputs', alphas)
            
        item_his_eb_new = tf.concat([self.item_his_eb, tf.expand_dims(self.time_last_his_batch_ph, -1)], -1)
        item_his_eb_new = tf.concat([item_his_eb_new, tf.expand_dims(self.time_now_his_batch_ph, -1)], -1)
        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, final_state1 = dynamic_rnn(Time4ALSTMCell(HIDDEN_SIZE), inputs=item_his_eb_new,
                                         att_scores = tf.expand_dims(alphas, -1),
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="lstm1")
            tf.summary.histogram('LSTM_outputs', rnn_outputs)

        inp = tf.concat([self.item_eb, final_state1[1]], 1)
        self.build_fcn_net(inp, use_dice=True)
        
class Model_TC_SeqRec(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_TC_SeqRec, self).__init__(n_uid, n_mid, n_cat,
                                                       EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                       use_negsampling)

        item_his_eb_new = tf.concat([self.item_his_eb, tf.expand_dims(self.time_last_his_batch_ph, -1)], -1)
        item_his_eb_new = tf.concat([item_his_eb_new, tf.expand_dims(self.time_now_his_batch_ph, -1)], -1)
        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, final_state1 = dynamic_rnn(Time4LSTMCell(HIDDEN_SIZE), inputs=item_his_eb_new,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="lstm1")
            tf.summary.histogram('LSTM_outputs', rnn_outputs)

        # Attention layer
        with tf.name_scope('Attention_layer'):
            att_outputs, alphas = din_fcn_attention(self.item_eb, rnn_outputs, ATTENTION_SIZE, self.mask,
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
            tf.summary.histogram('alpha_outputs', alphas)
            att_fea = tf.reduce_sum(att_outputs, 1)
                
        inp = tf.concat([self.item_eb, att_fea], 1)
        self.build_fcn_net(inp, use_dice=True)

class Model_SLi_Rec_Fixed(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_SLi_Rec_Fixed, self).__init__(n_uid, n_mid, n_cat,
                                                       EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                       use_negsampling)
        # Attention layer 1
        with tf.name_scope('Attention_layer_1'):
            att_outputs1, alphas1 = attention_HAN(self.item_his_eb, attention_size=ATTENTION_SIZE, return_alphas=True)
            att_fea1 = tf.reduce_sum(att_outputs1, 1)
            tf.summary.histogram('att_fea1', att_fea1)
            
        item_his_eb_new = tf.concat([self.item_his_eb, tf.expand_dims(self.time_last_his_batch_ph, -1)], -1)
        item_his_eb_new = tf.concat([item_his_eb_new, tf.expand_dims(self.time_now_his_batch_ph, -1)], -1)
        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, final_state1 = dynamic_rnn(Time4LSTMCell(HIDDEN_SIZE), inputs=item_his_eb_new,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="lstm1")
            tf.summary.histogram('LSTM_outputs', rnn_outputs)

        # Attention layer 2
        with tf.name_scope('Attention_layer_2'):
            att_outputs2, alphas2 = din_fcn_attention(self.item_eb, rnn_outputs, ATTENTION_SIZE, self.mask,
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
            tf.summary.histogram('alpha_outputs2', alphas2)
            att_fea2 = tf.reduce_sum(att_outputs2, 1)
                 
        #alpha    
        with tf.name_scope('User_alpha'):    
            user_alpha = 0.2 
            user_embed = att_fea1 * user_alpha + att_fea2 * (1.0 - user_alpha)
            
        inp = tf.concat([self.item_eb, user_embed], 1)
        self.build_fcn_net(inp, use_dice=True)
        
class Model_SLi_Rec_Adaptive(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_SLi_Rec_Adaptive, self).__init__(n_uid, n_mid, n_cat,
                                                       EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                       use_negsampling)
        # Attention layer 1
        with tf.name_scope('Attention_layer_1'):
            att_outputs1, alphas1 = attention_HAN(self.item_his_eb, attention_size=ATTENTION_SIZE, return_alphas=True)
            att_fea1 = tf.reduce_sum(att_outputs1, 1)
            tf.summary.histogram('att_fea1', att_fea1)
            
        item_his_eb_new = tf.concat([self.item_his_eb, tf.expand_dims(self.time_last_his_batch_ph, -1)], -1)
        item_his_eb_new = tf.concat([item_his_eb_new, tf.expand_dims(self.time_now_his_batch_ph, -1)], -1)
        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, final_state1 = dynamic_rnn(Time4LSTMCell(HIDDEN_SIZE), inputs=item_his_eb_new,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="lstm1")
            tf.summary.histogram('LSTM_outputs', rnn_outputs)

        # Attention layer 2
        with tf.name_scope('Attention_layer_2'):
            att_outputs2, alphas2 = din_fcn_attention(self.item_eb, rnn_outputs, ATTENTION_SIZE, self.mask,
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True, scope='1')
            tf.summary.histogram('alpha_outputs2', alphas2)
            att_fea2 = tf.reduce_sum(att_outputs2, 1)
         
        #alpha           
        with tf.name_scope('User_alpha'):    
            concat_all = tf.concat([self.item_eb, att_fea1, att_fea2, tf.expand_dims(self.time_now_his_batch_ph[:,-1], -1)], 1)
            concat_att1 = tf.layers.dense(concat_all, 80, activation=tf.nn.sigmoid, name='concat_att1')
            concat_att2 = tf.layers.dense(concat_att1, 40, activation=tf.nn.sigmoid, name='concat_att2')
            user_alpha = tf.layers.dense(concat_att2, 1, activation=tf.nn.sigmoid, name='concat_att3') 
            user_embed = att_fea1 * user_alpha + att_fea2 * (1.0 - user_alpha)

        inp = tf.concat([self.item_eb, user_embed], 1)
        self.build_fcn_net(inp, use_dice=True)