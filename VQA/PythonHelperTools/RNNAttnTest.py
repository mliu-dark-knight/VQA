import os

import numpy as np
from keras.callbacks import *
from keras.layers import *
from keras.layers import Dense
from keras.layers import Embedding
# import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from scipy import misc
from vqaTools.vqa import VQA

BATCH_SIZE = 512

dataDir = '../../VQA'
taskType = 'OpenEnded'
dataType = 'mscoco'  # 'mscoco' for real and 'abstract_v002' for abstract
dataSubType = 'train2014'
annFile = '%s/Annotations/%s_%s_annotations.json' % (dataDir, dataType, dataSubType)
quesFile = '%s/Questions/%s_%s_%s_questions.json' % (dataDir, taskType, dataType, dataSubType)
imgDir = '%s/Images/%s/%s/' % (dataDir, dataType, dataSubType)

BASE_DIR = ''
GLOVE_DIR = BASE_DIR + 'glove.6B/'
TEXT_DATA_DIR = BASE_DIR + '20_newsgroup/'
MAX_SEQUENCE_LENGTH = 50
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

# initialize VQA api for QA annotations

# load and display QA annotations for given images
"""
Usage: vqa.getImgIds(quesIds=[], quesTypes=[], ansTypes=[])
Above method can be used to retrieve imageIds for given question Ids or given question types or given answer types.
"""


def gentor():
    questions = []
    imagearr = np.empty((BATCH_SIZE, 200, 200, 3))
    labels = np.empty(BATCH_SIZE)
    i = 0

    while True:
        for ann in anns:
            imgId = ann['image_id']
            imgFilename = 'COCO_' + dataSubType + '_' + str(imgId).zfill(12) + '.jpg'
            fname = imgDir + imgFilename
            image = misc.imresize(misc.imread(fname, mode='RGB'), (200, 200, 3), 'bicubic')
            imagearr[i] = image / 255
            questions.append(vqa.qqa[ann['question_id']]['question'])
            if ann['answers'][0]['answer'] == 'yes':
                labels[i] = 1
            elif ann['answers'][0]['answer'] == 'no':
                labels[i] = 0
            else:
                continue
            i += 1
            if (i == BATCH_SIZE):
                # [tokenizer.texts_to_sequences(questions), np.float32(imagearr)]
                yield (data, to_categorical(labels, nb_classes=2))
                i = 0
                questions = []


# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')

texts = []
labels = []
vqa = VQA(annFile, quesFile)
ids = vqa.getImgIds(ansTypes='yes/no')
annIds = vqa.getQuesIds(imgIds=ids)
anns = vqa.loadQA(annIds)
for ann in anns:
    if ann['answers'][0]['answer'] == 'yes':
        labels.append(1)
        texts.append(vqa.qqa[ann['question_id']]['question'])
    elif ann['answers'][0]['answer'] == 'no':
        labels.append(0)
        texts.append(vqa.qqa[ann['question_id']]['question'])
    else:
        print(ann['answers'][0]['answer'] + " " + vqa.qqa[ann['question_id']]['question'])

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

print('Preparing embedding matrix.')

# prepare embedding matrix
nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= nb_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

modeltxt = Sequential()
modeltxt.add(
    Embedding(nb_words, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))
modeltxt.add(Bidirectional(
    GRU(1000, consume_less='gpu', input_length=MAX_SEQUENCE_LENGTH, dropout_W=0.5, dropout_U=0.2)))

# model = Sequential()
# model.add(Convolution2D(96, 11, 11, init="he_normal", border_mode='valid', subsample=(4, 4), input_shape=(200, 200, 3)))
# model.add(BatchNormalization())
# model.add(Activation("relu"))
#
# model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same'))
#
# model.add(Convolution2D(256, 5, 5, init="he_normal", border_mode='same'))
# model.add(BatchNormalization())
# model.add(Activation("relu"))
#
# model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same'))
#
# model.add(Convolution2D(384, 3, 3, init="he_normal", border_mode='same'))
# model.add(BatchNormalization())
# model.add(Activation("relu"))
#
# model.add(Convolution2D(384, 3, 3, init="he_normal", border_mode='same'))
# model.add(BatchNormalization())
# model.add(Activation("relu"))
#
# model.add(Convolution2D(384, 3, 3, init="he_normal", border_mode='same'))
# model.add(BatchNormalization())
# model.add(Activation("relu"))
#
# model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same'))
#
# model.add(Flatten())
modeltxt.add(Dense(100, init="he_normal", activation="relu"))
modeltxt.add(Dropout(0.5))
modeltxt.add(Dense(100, init="he_normal", activation="relu"))
modeltxt.add(Dropout(0.5))
modeltxt.add(Dense(2, init="glorot_normal", activation="softmax"))
print(modeltxt.summary())
modeltxt.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
# modeltxt.load_weights("weights.09.hdf5")

# modeltxt.fit_generator(gentor(), 100000, 10, verbose=1,
#                     callbacks=[ModelCheckpoint("weights.{epoch:02d}.hdf5", period=1)],
#                     pickle_safe=False)

modeltxt.fit(x_train, y_train, validation_data=(x_val, y_val), verbose=1,
             nb_epoch=1000, batch_size=1024, callbacks=[ModelCheckpoint("weights.{epoch:02d}.hdf5", period=1)])
