import os
import pickle
from collections import defaultdict
from multiprocessing import Queue, Process
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from VQA.PythonHelperTools.vqaTools.vqa import VQA


dataDir = 'VQA'
taskType = 'OpenEnded'
dataType = 'mscoco'
dataSubTypeTrain = 'train2014'
annFileTrain = '%s/Annotations/%s_%s_annotations.json' % (dataDir, dataType, dataSubTypeTrain)
quesFileTrain = '%s/Questions/%s_%s_%s_questions.json' % (dataDir, taskType, dataType, dataSubTypeTrain)
imgDirTrain = '%s/Images/%s/%s/' % (dataDir, dataType, dataSubTypeTrain)
featDirTrain = '%s/Features/%s/%s/' % (dataDir, dataType, dataSubTypeTrain)

dataSubTypeVal = 'val2014'
annFileVal = '%s/Annotations/%s_%s_annotations.json' % (dataDir, dataType, dataSubTypeVal)
quesFileVal = '%s/Questions/%s_%s_%s_questions.json' % (dataDir, taskType, dataType, dataSubTypeVal)
imgDirVal = '%s/Images/%s/%s/' % (dataDir, dataType, dataSubTypeVal)
featDirVal = '%s/Features/%s/%s/' % (dataDir, dataType, dataSubTypeVal)


class DataSet:
	def __init__(self, word2vec, params, type, num_threads=1, q_max=1):
		assert params.dataset_size is None or params.batch_size <= params.dataset_size, 'batch size cannot be greater than data size.'
		assert type == 'train' or type == 'val', 'bad data type'
		assert num_threads > 0, 'lol no threads'
		self.type = type
		self.batch_size = params.batch_size
		self.dataset_size = params.dataset_size
		self.max_ques_size = params.max_ques_size
		self.word2vec = word2vec
		if (self.type == 'train'):
			self.vqa = VQA(annFileTrain, quesFileTrain)
		elif (self.type == 'val'):
			self.vqa = VQA(annFileVal, quesFileVal)
		self.anns = self.load_QA()
		self.q_max = q_max
		self.queue = Queue(maxsize=self.q_max)
		self.counter = 0
		self.num_threads = num_threads
		self.start()

	def start(self):
		self.process_list = []
		for i in range(0, self.num_threads):
			self.process_list.append(Process(target=self.next_batch_thread))

		for proc in self.process_list:
			proc.start()

	def kill(self):
		for proc in self.process_list:
			proc.terminate()

	def load_QA(self):
		annIds = self.vqa.getQuesIds()
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
		self.counter += 1
		if (self.counter % 1000 == 0):
			self.kill()
			del self.queue
			self.queue = Queue(maxsize=self.q_max)
			self.start()

		return self.queue.get()

	def next_batch_thread(self, visualize=False):
		while True:
			Anns, Is, Xs, Qs, As = {'b': [], 'm': []}, {'b': [], 'm': []}, {'b': [], 'm': []}, {'b': [], 'm': []}, {
				'b': [],
				'm': []}
			for randomAnn in np.random.choice(self.anns, size=self.batch_size):
				imgId = randomAnn['image_id']
				if (self.type == 'train'):
					imgFilename = 'COCO_' + dataSubTypeTrain + '_' + str(imgId).zfill(12) + '.jpg'
					featFilename = 'COCO_' + dataSubTypeTrain + '_' + str(imgId).zfill(12) + '.npy'
				elif (self.type == 'val'):
					imgFilename = 'COCO_' + dataSubTypeVal + '_' + str(imgId).zfill(12) + '.jpg'
					featFilename = 'COCO_' + dataSubTypeVal + '_' + str(imgId).zfill(12) + '.npy'

				try:
					if (self.type == 'train'):
						I, X = io.imread(imgDirTrain + imgFilename), np.load(featDirTrain + featFilename)
					elif (self.type == 'val'):
						I, X = io.imread(imgDirVal + imgFilename), np.load(featDirVal + featFilename)

					Q = np.stack(
						[self.word2vec.word_vector(word) for word in self.id_to_question(randomAnn['question_id'])])
					A = self.word2vec.one_hot(self.id_to_answer(randomAnn['question_id']))
				except:
					continue
				if randomAnn['answer_type'] == 'yes/no':
					type = 'b'
					A = 0 if self.id_to_answer(randomAnn['question_id']) == 'no' else 1
				else:
					type = 'm'
				if visualize:
					self.visualize(randomAnn, I)
				Anns[type].append(randomAnn)
				Is[type].append(I)
				Xs[type].append(X)
				Qs[type].append(Q)
				As[type].append(A)

			self.queue.put(
				(np.array(Anns['b']), np.array(Is['b']), np.array(Xs['b']), np.array(Qs['b']), np.array(As['b']),
				 np.array(Anns['m']), np.array(Is['m']), np.array(Xs['m']), np.array(Qs['m']), np.array(As['m'])))


class WordTable:
	def __init__(self, dim):
		self.word2vec = self.load_glove(dim)
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

	def load_glove(self, dim):
		word2vec = {}

		path = "VQA/PythonHelperTools/glove.6B/glove.6B." + str(dim) + 'd'
		try:
			with open(path + '.cache', 'rb') as cache_file:
				word2vec = pickle.load(cache_file)

		except:
			# Load n create cache
			with open(path + '.txt') as f:
				for line in f:
					l = line.split()
					word2vec[l[0]] = [float(x) for x in l[1:]]

			with open(path + '.cache', 'wb') as cache_file:
				pickle.dump(word2vec, cache_file)

		print("Glove data loaded")
		return word2vec
