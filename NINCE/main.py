from collections import defaultdict
from keras.layers import Input, Dense, Dropout, Embedding, GlobalAveragePooling1D, GRU, Bidirectional
from keras.layers import GlobalMaxPooling1D, LSTM, Dropout, SimpleRNN, TimeDistributed
from keras.models import Model, Sequential
from keras.optimizers import Adam
# from keras.engine.topology import Layer
from keras.layers import concatenate
from keras import activations, initializers, constraints
from keras import regularizers
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
import keras.backend as K
import numpy as np
from keras.utils import pad_sequences

import os
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import tensorflow as tf

import re
import spacy
from transformers import RobertaTokenizer, RobertaModel  # 修改这里
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from transformers import XLNetModel,XLNetConfig,XLNetTokenizer
from collections import Counter

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# tf.keras.backend.set_session(sess)

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#             tf.config.experimental.set_virtual_device_configuration(
#                 gpu,
#                 [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 2)]) # 这里以 2GB 为例
#     except RuntimeError as e:
#         print(e)

from layers import *
from utils import *



def NINCE(GCNXss_shape, GCNXpp_shape, n_head=8, size_per_head=8, MAX_REV_LEN=75, MAX_REV_WORD_LEN=1, support=3):
    
    '''
    Comment Encoding
    '''
    
    ''' Capture reviews context correlation'''
    ## word-level encoding
    word_input = Input(shape=(None, 768), dtype='float32')
    word_sa = Self_Attention(n_head, size_per_head)(word_input)
    word_avg = GlobalAveragePooling1D()(word_sa)
    wordEncoder = Model(word_input, word_avg)
    
    ## review-level encoding
    content_input = Input(shape=(MAX_REV_LEN, MAX_REV_WORD_LEN, 768), dtype='float32')
    content_word_encode = TimeDistributed(wordEncoder, name='word_seq_encoder')(content_input)
    content_sa = Self_Attention(n_head, size_per_head)(content_word_encode)
    contentSA_avg_pool = GlobalAveragePooling1D()(content_sa) # session embedding
    
    ''' Capture Post-Comment co-attention'''
    post_words_input = Input(shape=(None, 768), dtype='float32')
    post_lstm = Bidirectional(GRU(32, return_sequences=True))(post_words_input)
    coAtt_vec = CoAttLayer(MAX_REV_LEN)([content_word_encode, post_lstm])
    
    '''
    GCN
    Session-Session Interaction Extractor
    Adjacency: session-session
    '''
    # G_ss = [Input(shape=(None, None), batch_shape=(None, None), sparse=True) for _ in range(3)]
    # test = Input(shape=(None, None))
    # G_ss = [test, test, test]
    G_ss = [Input(shape=(None,), sparse=True) for _ in range(3)]
    X_ss = Input(shape=(GCNXss_shape,))
    X_ss_emb = Dense(16, activation='relu')(X_ss)
    
    # Define GCN model architecture
    H_ss = Dropout(0.2)(X_ss_emb)
    print(f'G_ss_shape: {G_ss[0].shape}')
    H_ss = GraphConvolution(16, support, activation='relu', kernel_regularizer=l2(5e-4))([H_ss]+G_ss)
    H_ss = GraphConvolution(8, support, activation='softmax', kernel_regularizer=l2(5e-4))([H_ss]+G_ss)
    
    '''
    GCN
    Post-Post Interaction Extractor
    Adjacency: post-post
    '''
    # G_pp = [Input(shape=(None, None), batch_shape=(None, None), sparse=True) for _ in range(3)]
    G_pp = [Input(shape=(None,), sparse=True) for _ in range(3)]
    X_pp = Input(shape=(GCNXpp_shape,))
    X_pp_emb = Dense(16, activation='relu')(X_pp)
    
    # Define GCN model architecture
    H_pp = Dropout(0.2)(X_pp_emb)
    H_pp = GraphConvolution(16, support, activation='relu', kernel_regularizer=l2(5e-4))([H_pp]+G_pp)
    H_pp = GraphConvolution(8, support, activation='softmax', kernel_regularizer=l2(5e-4))([H_pp]+G_pp)

    '''
    Concatenate Comment Encoding & GCN Embedding
    '''
    H = concatenate([contentSA_avg_pool, coAtt_vec, H_ss, H_pp])
    Y = Dense(1, activation='sigmoid')(H)
    
    # Compile model
    model = Model(inputs=[content_input]+[post_words_input]+[X_ss]+G_ss+[X_pp]+G_pp, outputs=Y)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.01))
    # model.summary()
    
    return model
'''
Load data
'''
# load preprocessed data
with open('preprocessData/Dat4Model_ins.pickle', 'rb') as f:
    Dat4Model1 = pickle.load(f)

with open('encoded_reviews_twoembeddings.pkl', 'rb') as file2:
    data = pickle.load(file2)


comment_n = 0
# 遍历 data
for i in range(len(data)):
    original_shape = data[i].shape
    n = original_shape[0]
    if n > comment_n:
        comment_n = n

for i in range(len(data)):
    # 获取原始 ndarry 的形状
    original_shape = data[i].shape

    padding_array = np.zeros((comment_n, original_shape[1], original_shape[2]))

    if original_shape[0] < comment_n:
        padding_array[:original_shape[0]] = data[i]
    else:
        padding_array = data[i][:comment_n]

    # 将填充后的数组重新赋值给 data
    data[i] = padding_array

# 检查第一个 ndarry 的形状
print(data[0].shape)
w2v_vec_all = np.array(data)

############

# load multi-hot user vectors of each session
with open('preprocessData/multi_hot_users_ins.pickle', 'rb') as f:
    multi_hot_users1 = pickle.load(f)  

y_all = Dat4Model1['y_all'] # target for HENIN
textFeat_all = Dat4Model1['textFeat_all']
# w2v_vec_all = Dat4Model['w2v_vec_all'] 

MAX_REV_WORD_LEN = w2v_vec_all.shape[2]
MAX_REV_LEN = w2v_vec_all.shape[1]
postEmb = pad_sequences(w2v_vec_all[:,0,:,:], maxlen=MAX_REV_LEN, dtype='float32', padding='post')
# post text embedding: the first element in second dimension is the post
# postEmb = pad_sequences(first_comment_embeddings, maxlen=MAX_REV_LEN, dtype='float32', padding='post')
print("postEmb.shape",postEmb.shape)
print("w2v_vec_all.shape",w2v_vec_all.shape)
# Initialize dictionaries to store results for each epoch
epoch_results_CP = defaultdict(list)
epoch_results_GP = defaultdict(list)

## cross validating for HENIN model

# 初始化一个全局的列表来存储所有折的所有测试结果
global_test_results = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}
# epoch_average_metrics = []

# 存储每一折的评估结果
losses = []
def NINCE_cv(graph, y, A, model, epochs):
    epoch_average_metrics = []  # Initialize epoch_average_metrics

    skf = StratifiedKFold(n_splits=5, random_state=9999, shuffle=True)
    iters = 0
    print(f'A_shape {A.shape[0]}')
    # n-fold cross validation
    for train_index, test_index in skf.split(range(len(y)), y):
        print(f'Fold {iters + 1}:')
        y_train, y_test, train_mask = Mask_y(y=y, train_ix=train_index, test_ix=test_index)
        clf = HENIN(GCNXss_shape=multi_hot_users1.shape[1],
                    GCNXpp_shape=textFeat_all[:, 0, :].shape[1],
                    n_head=8, size_per_head=8, MAX_REV_LEN=MAX_REV_LEN,
                    MAX_REV_WORD_LEN=MAX_REV_WORD_LEN, support=3
                    )
        # clf = model
        fold_epoch_metrics = {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'auc': []}
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}:')
            clf.fit(graph, y_train, sample_weight=train_mask, batch_size=A.shape[0], epochs=1)
            preds = (clf.predict(graph, batch_size=A.shape[0])[:,0] >= 0.5).astype(int)
            completePerform = metrics(y, preds) # complete set performance
            print(f'Performance of Fold {iters}:', completePerform)
            generalPerform = metrics(y[test_index], preds[test_index]) # test set performance
            print(f'test Performance of Fold {iters}:', generalPerform)


            # try:
            #     CP = {k: v + [completePerform[k]] for k, v in CP.items()}
            #     print(f'CP: {CP}')
            #     GP = {k: v + [generalPerform[k]] for k, v in GP.items()}
            #     print(f'GP: {GP}')
            # except:
            #     CP = completePerform
            #     print(f'completePerform  CP: {CP}')
            #     GP = generalPerform
            #     print(f'generalPerform  GP: {GP}')
            # iters += 1
            for key in generalPerform:
                fold_epoch_metrics[key].append(generalPerform[key])
        # iters += 1
        if len(epoch_average_metrics) == 0:
            # 如果是第一折，直接将结果复制过来
            epoch_average_metrics = fold_epoch_metrics
        else:
            # 否则，累加每个epoch的结果，稍后计算平均值
            for key in epoch_average_metrics:
                epoch_average_metrics[key] = [sum(x) for x in zip(epoch_average_metrics[key], fold_epoch_metrics[key])]
    # Calculate average results for each epoch
    # AvgCP = {k: '{:.3f}'.format(np.mean(v)) for k, v in epoch_results_CP.items()}
    # AvgGP = {k: '{:.3f}'.format(np.mean(v)) for k, v in epoch_results_GP.items()}
        iters = iters + 1
    # 计算每个epoch的平均性能指标
    for epoch in range(epochs):
        for key in epoch_average_metrics:
            if epoch < len(epoch_average_metrics[key]):
                epoch_average_metrics[key][epoch] /= 5
            else:
                break

    # 写入到文件中
    with open('data/avg_test_result1.txt', 'w') as f:
        for epoch in range(epochs):
            epoch_results = {key: f"{epoch_average_metrics[key][epoch]:.4f}" for key in epoch_average_metrics}
            f.write(f"Epoch {epoch + 1}: {epoch_results}\n")
        # 再将auc最大的那个epoch的结果写入到文件中
        max_auc_epoch = np.argmax(epoch_average_metrics['auc'])
        epoch_results = {key: f"{epoch_average_metrics[key][max_auc_epoch]:.4f}" for key in epoch_average_metrics}

    # print(f'Average CP: {AvgCP}')
    # print(f'Average GP: {AvgGP}')
    return AvgCP, AvgGP

OurComResult = {}
OurGenResult = {}

ppA = genAdjacencyMatrix(textFeat_all[:,0,:], 'cosine')
ssA = genAdjacencyMatrix(multi_hot_users1, 'cosine')
batch_size = ppA.shape[0]
graph_ss = genGCNgraph(ssA, multi_hot_users1)
graph_pp = genGCNgraph(ppA, textFeat_all[:,0,:])

graph = [w2v_vec_all]+[postEmb]+graph_ss+graph_pp
# print("w2v_vec_all.shape",w2v_vec_all.shape)
# print("postEmb.shape",postEmb.shape)

clf = NINCE(GCNXss_shape=multi_hot_users1.shape[1],
            GCNXpp_shape=textFeat_all[:, 0, :].shape[1],
            n_head=8, size_per_head=8, MAX_REV_LEN=MAX_REV_LEN,
            MAX_REV_WORD_LEN=MAX_REV_WORD_LEN, support=3)

AvgCP, AvgGP = NINCE_cv(graph=graph, y=y_all, A=ppA, model=clf, epochs=80)
print(AvgGP)

