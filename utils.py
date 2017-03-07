import os
import random
import pickle
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from collections import defaultdict
from VQA.PythonHelperTools.vqaTools.vqa import VQA

dataDir = 'VQA'
taskType = 'OpenEnded'
dataType = 'mscoco'
dataSubType = 'val2014'
annFile = '%s/Annotations/%s_%s_annotations.json' % (dataDir, dataType, dataSubType)
quesFile = '%s/Questions/%s_%s_%s_questions.json' % (dataDir, taskType, dataType, dataSubType)
imgDir = '%s/Images/%s/%s/' % (dataDir, dataType, dataSubType)
featDir = '%s/Features/%s/%s/' %(dataDir, dataType, dataSubType)
annFile = '%s/Annotations/%s_%s_annotations.json' % (dataDir, dataType, dataSubType)
quesFile = '%s/Questions/%s_%s_%s_questions.json' % (dataDir, taskType, dataType, dataSubType)


class DataSet:
    def __init__(self, word2vec, params):
        assert params.dataset_size is None or params.batch_size <= params.dataset_size, 'batch size cannot be greater than data size.'
        self.batch_size = params.batch_size
        self.dataset_size = params.dataset_size
        self.max_ques_size = params.max_ques_size
        self.word2vec = word2vec
        self.vqa = VQA(annFile, quesFile)
        self.anns = self.load_QA()

    def load_QA(self):
        annIds = self.vqa.getQuesIds(imgIds=[42, 73, 74, 133, 136, 139, 143, 164, 192, 196])
        if self.dataset_size is not None:
            annIds = annIds[:self.dataset_size]
        return self.vqa.loadQA(annIds)

    def id_to_question(self, id=None):
        question = self.vqa.qqa[id]['question'][:-1].split()
        return [None] * (self.max_ques_size - len(question)) + list(map(lambda str: str.lower(), question))

    def id_to_answer(self, id=None):
        ans_dict = defaultdict(lambda: 0)
        for answer in self.vqa.loadQA(id)[0]['answers']:
            if len(answer['answer'].split()) == 1:
                ans_dict[answer['answer']] += 1
        return max(ans_dict, key=lambda k: ans_dict[k])

    def index_to_word(self, index):
        return self.word2vec.index_to_word(index)

    def visualize(self, ann, I):
        self.vqa.showQA([ann])
        plt.imshow(I)
        plt.axis('off')
        plt.show()

    def next_batch(self, visualize=False):
        Anns, Is, Xs, Qs, As = [], [], [], [], []
        while len(Is) < self.batch_size:
            randomAnn = random.choice(self.anns)
            imgId = randomAnn['image_id']
            imgFilename = 'COCO_' + dataSubType + '_'+ str(imgId).zfill(12) + '.jpg'
            featFilename = 'COCO_' + dataSubType + '_' + str(imgId).zfill(12) + '.npy'
            try:
                I, X = io.imread(imgDir + imgFilename), np.load(featDir + featFilename)
                Q = np.stack([self.word2vec.word_vector(word) for word in self.id_to_question(randomAnn['question_id'])])
                A = self.word2vec.one_hot(self.id_to_answer(randomAnn['question_id']))
            except:
                continue
            Anns.append(randomAnn)
            Is.append(I)
            Xs.append(X)
            Qs.append(Q)
            As.append(A)
            if visualize:
                self.visualize(randomAnn, I)

        assert len(Is) == len(Xs) and len(Xs) == len(Qs) and len(Qs) == len(As)
        assert type(np.array(Qs)) == np.ndarray and type(Qs[0]) == np.ndarray
        return (np.array(Anns), np.array(Is), np.stack(Xs), np.array(Qs), np.stack(As))


class WordTable:
    def __init__(self, dim):
        self.word2vec = WordTable.load_glove(dim)
        self.dim = dim
        self.vocab_size = len(self.word2vec)
        self.index_word()

    def index_word(self):
        self.word2idx = {}
        self.idx2word = {}
        for idx, word in enumerate(self.word2vec):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def word_vector(self, word):
        if word == None:
            return np.zeros(self.dim)
        return self.word2vec[word]

    # for sparse softmax cross entropy
    def one_hot(self, word):
        return self.word2idx[word]

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

