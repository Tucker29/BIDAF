import numpy as np
import os

'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' #默认值，输出所有信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #屏蔽通知信息（INFO）
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #屏蔽通知信息和警告信息（INFO\WARNING）
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #屏蔽通知信息、警告信息和报错信（INFO\WARNING\FATAL）
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 忽略警告消息
import warnings
warnings.filterwarnings('ignore')   # action=“ignore”	忽略匹配的警告

import logging
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

# 自定义的程序
import layers       #./BiDAF_tf2/layers
import preprocess   #./BiDAF_tf2

print("tf.__version__:", tf.__version__)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print("============?",e)

class BiDAF:

    def __init__(
            self, clen, qlen, max_char_len,emb_size,
            max_features=5000,      # 字典内的单词数/特征数
            num_highway_layers=2,
            encoder_dropout=0,
            num_decoders=2,
            decoder_dropout=0,
            vocab_size,
            embedding_matrix,
            conv_layers=[]
    ):

        self.clen = clen
        self.qlen = qlen
        self.max_char_len = max_char_len
        self.max_features = max_features
        self.emb_size = emb_size
        self.vocab_size = vocab_size
        self.embedding_matrix = embedding_matrix
        self.conv_layers = conv_layers
        self.num_highway_layers = num_highway_layers
        self.encoder_dropout = encoder_dropout
        self.num_decoders = num_decoders
        self.decoder_dropout = decoder_dropout

    def build_model(self):
        """
        构建模型
        :return:
        """
        # 1 embedding 层
        # TODO：homework：使用glove word embedding（或自己训练的w2v） 和 CNN char embedding
        '''
        layers.Input(
                shape=None,
                batch_size=None,
                name=None,
                dtype=None,
                sparse=False,
                tensor=None,
                ragged=False,
                **kwargs,
                )
        # Word Embedding Layer
        # 用GloVe初始化 word embedding
        '''
        cinn_c = tf.keras.layers.Input(shape=(self.clen,), self.max_char_len,name='context_input_char')
        qinn_c = tf.keras.layers.Input(shape=(self.qlen,), self.max_char_lenname='question_input_char')
        embedding_layer_char = tf.keras.layers.Embedding(self.max_features,  # 词汇表最大数量
                                                         self.emb_size,  # 词向量维度
                                                         embeddings_initializer='uniform',
                                                         )
        '''
        input_dim	词汇表的维度(总共有多少个不相同的词)
        output_dim	嵌入词空间的维度
        input_length	输入语句的长度
        embeddings_initializer	
        embeddings_regularizer	
        embeddings_constraint	
        mask_zero

        输入形状:二维张量(batch_size,input_length)
        输出形状:三维张量(batch_size,input_length,output_dim)
        '''
        # 经过embedding得到emb_cc ，emb_cq
        # Char Embedding Layer：通过character-level的CNNs，把每个词映射到向量空间
        emb_cc = embedding_layer_char(cinn_c)
        emb_cq = embedding_layer_char(qinn_c)

        # 对c,q分别进行卷积和池化
        c_conv_out = []
        q_conv_out = []

        filter_sizes = sum(list(np.array(self.conv_layers).T[0]))
        assert filter_sizes == self.emb_size

        # 卷积
        # filters：卷积过滤器的数量,对应输出的维数
        # kernel_size：整数,过滤器的大小,如果为一个整数，则宽和高相同
        # strides：横向和纵向的步长,如果为一个整数，则横向和纵向相同
        # activation：激活函数,None是线性函数
        # padding：same表示不够卷积核大小的块就补0,所以输出和输入形状相同。（valid:表示不够卷积核大小的块,则丢弃）
        for filters, kernel_size in self.conv_layers:
            conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=[kernel_size, self.emb_size], strides=1,
                                          activation='relu',padding='same')(emb_cc)
            conv = tf.reduce_max(conv, 2)  # 池化
            c_conv_out.append(conv)

            conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=[kernel_size, self.emb_size], strides=1,
                                        activation='relu',padding='same')(emb_cq)
            conv = tf.reduce_max(conv, 2)  # 池化
            q_conv_out.append(conv)
        c_conv_out = tf.keras.layers.concatanate(c_conv_out)
        q_conv_out = tf.keras.layers.concatanate(q_conv_out)

        cinn_w = tf.keras.layers.Input(shape=(self.clen,), name='context_input_word')
        qinn_w = tf.keras.layers.Input(shape=(self.qlen,), name='question_input_word')

        # Word Embedding Layer：通过训练好的word embedding将每个词映射到一个向量空间。
        embedding_layer_word = tf.keras.layers.Embedding(
            self.vocab_size,    # 在此是词典大小
            self.emb_size,
            embeddings_initializer=tf.keras.layers_initializer(self.embedding_matrix), trainable=False
        )

        emb_cw = embedding_layer_word(cinn_w)
        emb_qw = embedding_layer_word(qinn_w)

        cemb = tf.concat([emb_cw, c_conv_out], axis = 2)  # tf.concatt把多个array沿着某一个维度拼接在一起，成100dim
        qemb = tf.concat([emb_qw, q_conv_out], axis = 2)

        for i in range(self.num_highway_layers):  # 高速神经网络的个数 2
            """
            使用两层高速神经网络
            """
            highway_layer = layers.Highway(name=f'Highway{i}')
            chighway = tf.keras.layers.TimeDistributed(highway_layer, name=f'CHighway{i}')
            qhighway = tf.keras.layers.TimeDistributed(highway_layer, name=f'QHighway{i}')
            cemb = chighway(cemb)
            qemb = qhighway(qemb)

        ## 2. 上下文嵌入层
        # 编码器 双向LSTM
        encoder_layer = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                self.emb_size,
                recurrent_dropout=self.encoder_dropout,
                return_sequences=True,
                name='RNNEncoder'
            ), name='BiRNNEncoder'
        )

        cencode = encoder_layer(cemb)  # 编码context
        qencode = encoder_layer(qemb)  # 编码question

        # 3.注意流层:结合query和context向量获得具有context中每个词具有query-aware特征的向量。
        similarity_layer = layers.Similarity(name='SimilarityLayer')
        similarity_matrix = similarity_layer([cencode, qencode])

        c2q_att_layer = layers.C2QAttention(name='C2QAttention') # Context-to-query 注意力计算
        q2c_att_layer = layers.Q2CAttention(name='Q2CAttention') # Query-to-context 注意力计算

        c2q_att = c2q_att_layer(similarity_matrix, qencode)
        q2c_att = q2c_att_layer(similarity_matrix, cencode)

        # 上下文嵌入向量的生成
        merged_ctx_layer = layers.MergedContext(name='MergedContext')
        merged_ctx = merged_ctx_layer(cencode, c2q_att, q2c_att)

        # 4.模型层:使用RNN扫描context
        modeled_ctx = merged_ctx
        for i in range(self.num_decoders):
            decoder_layer = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    self.emb_size,
                    recurrent_dropout=self.decoder_dropout,
                    return_sequences=True,
                    name=f'RNNDecoder{i}'
                ), name=f'BiRNNDecoder{i}'
            )
            modeled_ctx = decoder_layer(merged_ctx)

        # 5. 输出层
        span_begin_layer = layers.SpanBegin(name='SpanBegin')
        span_begin_prob = span_begin_layer([merged_ctx, modeled_ctx])

        span_end_layer = layers.SpanEnd(name='SpanEnd')
        span_end_prob = span_end_layer([cencode, merged_ctx, modeled_ctx, span_begin_prob])

        output_layer = layers.Combine(name='CombineOutputs')
        out = output_layer([span_begin_prob, span_end_prob])

        inn = [cinn, qinn]

        self.model = tf.keras.models.Model(inn, out)
        self.model.summary(line_length=128)

        optimizer = tf.keras.optimizers.Adadelta(lr=1e-2)
        self.model.compile(optimizer=optimizer,loss=negative_avg_log_error,metrics=[accuracy])

def negative_avg_log_error(y_true, y_pred):
    """
    损失函数计算
    -1/N{sum(i~N)[(log(p1)+log(p2))]}
    :param y_true:
    :param y_pred:
    :return:
    """

    def sum_of_log_prob(inputs):
        y_true, y_pred_start, y_pred_end = inputs

        begin_idx = tf.dtypes.cast(y_true[0], dtype=tf.int32)
        end_idx = tf.dtypes.cast(y_true[1], dtype=tf.int32)

        begin_prob = y_pred_start[begin_idx]
        end_prob = y_pred_end[end_idx]

        return tf.math.log(begin_prob) + tf.math.log(end_prob)

    y_true = tf.squeeze(y_true)
    y_pred_start = y_pred[:, 0, :]
    y_pred_end = y_pred[:, 1, :]

    inputs = (y_true, y_pred_start, y_pred_end)
    batch_prob_sum = tf.map_fn(sum_of_log_prob, inputs, dtype=tf.float32)

    return -tf.keras.backend.mean(batch_prob_sum, axis=0, keepdims=True)


def accuracy(y_true, y_pred):
    """
    准确率计算
    :param y_true:
    :param y_pred:
    :return:
    """

    def calc_acc(inputs):
        y_true, y_pred_start, y_pred_end = inputs

        begin_idx = tf.dtypes.cast(y_true[0], dtype=tf.int32)
        end_idx = tf.dtypes.cast(y_true[1], dtype=tf.int32)

        start_probability = y_pred_start[begin_idx]
        end_probability = y_pred_end[end_idx]

        return (start_probability + end_probability) / 2.0

    y_true = tf.squeeze(y_true)
    y_pred_start = y_pred[:, 0, :]
    y_pred_end = y_pred[:, 1, :]

    inputs = (y_true, y_pred_start, y_pred_end)
    acc = tf.map_fn(calc_acc, inputs, dtype=tf.float32)

    return tf.math.reduce_mean(acc, axis=0)

if __name__ == '__main__':
    tmp_path="D:/dhm/programer-lx/BiDAF_tf2"
    ds = preprocess.Preprocessor([
        tmp_path+'/data/squad/train-v1.1.json',
        tmp_path+'/data/squad/dev-v1.1.json',
        tmp_path+'/data/squad/dev-v1.1.json'
    ])

##    train_c, train_q, train_y = ds.get_dataset(tmp_path+'/data/squad/train-v1.1.json')
##    test_c, test_q, test_y = ds.get_dataset(tmp_path+'/data/squad/dev-v1.1.json')
    train_cc, train_cq, train_wc, train_wq, train_y = ds.get_dataset(tmp_path+'/data/squad/test.json')
    test_cc, test_cq, test_wc, test_wq, test_y = ds.get_dataset(tmp_path+'/data/squad/test.json')

    bidaf = BiDAF(
        clen=ds.max_clen,
        qlen=ds.max_qlen,
        emb_size=50,
        max_features=len(ds.charset),  # ds.charset
        vocab_size=len(ds.word_list),
        conv_layers=[[10, 1], [10, 2], [30, 3]],  # 卷积的大小及个数
        embedding_matrix = ds.embedding_matrix,   #glove中的
        max_char_len=ds.max_char_len
    )
    bidaf.build_model()
    bidaf.model.fit(
        [train_cc, train_qc, train_cw, train_qw], train_y,
        batch_size=64,
        epochs=10,
        validation_data=([test_cc, test_qc, test_cw, test_qw], test_y)
    )
else:
    print('__name__=======',__name__)