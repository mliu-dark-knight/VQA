import os
import pickle
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from VQA.PythonHelperTools.vqaTools.vqa import VQA

dataDir = 'VQA'
taskType = 'OpenEnded'
dataType = 'mscoco' # 'mscoco' for real and 'abstract_v002' for abstract
dataSubType = 'val2014'
annFile = '%s/Annotations/%s_%s_annotations.json' % (dataDir, dataType, dataSubType)
quesFile = '%s/Questions/%s_%s_%s_questions.json' % (dataDir, taskType, dataType, dataSubType)
imgDir = '%s/Images/%s/%s/' % (dataDir, dataType, dataSubType)
featDir = '%s/Features/%s/%s/' %(dataDir, dataType, dataSubType)
annFile = '%s/Annotations/%s_%s_annotations.json' % (dataDir, dataType, dataSubType)
quesFile = '%s/Questions/%s_%s_%s_questions.json' % (dataDir, taskType, dataType, dataSubType)

class DataSet:
    def __init__(self, word2vec, batch_size, dataset_size=None):
        assert batch_size <= dataset_size, 'batch size cannot be greater than data size.'
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.word2vec = word2vec
        self.vqa = VQA(annFile, quesFile)
        self.anns = self.load_QA()

    def load_QA(self):
        annIds = self.vqa.getQuesIds(ansTypes='yes/no')
        if self.dataset_size is not None:
            annIds = annIds[:self.dataset_size]
        return self.vqa.loadQA(annIds)

    def id_to_question(self, id=None):
        return map(lambda str: str.lower(), self.vqa.qqa[id]['question'][:-1].split())

    def id_to_answer(self, id=None):
        return self.vqa.loadQA(id)[0]['answers'][0]['answer'].lower()

    def next_batch(self):
        randomAnns = np.random.choice(self.anns, self.batch_size)
        # self.vqa.showQA(randomAnns)
        Is, Xs, Qs, As = [], [], [], []
        for randomAnn in randomAnns:
            imgId = randomAnn['image_id']
            imgFilename = 'COCO_' + dataSubType + '_'+ str(imgId).zfill(12) + '.jpg'
            featFilename = 'COCO_' + dataSubType + '_' + str(imgId).zfill(12) + '.npz'

            Is.append(io.imread(imgDir + imgFilename))
            Xs.append(np.load(featDir + featFilename))
            Qs.append(np.stack([self.word2vec.word_vector(word) for word in self.id_to_question(randomAnn['question_id'])]))
            As.append(self.id_to_answer(randomAnn['question_id']))
        return Is, np.stack(Xs), Qs, np.stack(As)

class WordTable:
    def __init__(self, dim):
        self.word2vec = WordTable.load_glove(dim)
        self.vocab_size = len(self.word2vec)
        self.index_word()

    def index_word(self):
        self.word2idx = {}
        self.idx2word = {}
        for idx, word in enumerate(self.word2vec):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def word_vector(self, word):
        return self.word2vec[word]

    def one_hot(self, word):
        vec = np.zeros(self.vocab_size)
        vec[self.word2idx[word]] = 1.0
        return vec

    def word_to_index(self, word):
        return self.word2idx[word]

    def index_to_word(self, index):
        return self.idx2word[index]

    @staticmethod
    def load_glove(dim):
        word2vec = {}

        path = "VQA/PythonHelperTools/glove.6B/glove.6B." + str(dim) + 'd'
        if os.path.exists(path + '.cache'):
            with open(path + '.cache', 'rb') as cache_file:
                word2vec = pickle.load(cache_file)

        else:
            # Load n create cache
            with open(path + '.txt') as f:
                for line in f:
                    l = line.split()
                    word2vec[l[0]] = [float(x) for x in l[1:]]

            with open(path + '.cache', 'wb') as cache_file:
                pickle.dump(word2vec, cache_file)

        print("Glove data loaded")
        return word2vec

