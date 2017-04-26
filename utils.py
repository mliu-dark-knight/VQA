import pickle
import threading
from queue import Queue
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
from VQA.PythonHelperTools.vqaTools.vqa import VQA

dataDir = 'VQA'
taskType = 'OpenEnded'
dataType = 'mscoco'
dataSubTypeTrain = 'val2014'
AnnoSubTypeTrain = 'val2014'
annFileTrain = '%s/Annotations/%s_%s_annotations.json' % (dataDir, dataType, AnnoSubTypeTrain)
quesFileTrain = '%s/Questions/%s_%s_%s_questions.json' % (dataDir, taskType, dataType, AnnoSubTypeTrain)
imgDirTrain = '%s/Images/%s/%s/' % (dataDir, dataType, dataSubTypeTrain)
featDirTrain = '%s/Features/%s/%s/' % (dataDir, dataType, dataSubTypeTrain)

dataSubTypeVal = 'val2014'
AnnoSubTypeVal = 'val2014'
annFileVal = '%s/Annotations/%s_%s_annotations.json' % (dataDir, dataType, AnnoSubTypeVal)
quesFileVal = '%s/Questions/%s_%s_%s_questions.json' % (dataDir, taskType, dataType, AnnoSubTypeVal)
imgDirVal = '%s/Images/%s/%s/' % (dataDir, dataType, dataSubTypeVal)
featDirVal = '%s/Features/%s/%s/' % (dataDir, dataType, dataSubTypeVal)


class DataSet:
	def __init__(self, word2vec, params, type, num_threads=1, q_max=1):
		assert params.dataset_size is None or params.batch_size <= params.dataset_size, 'batch size cannot be greater than data size.'
		assert type == 'train' or type == 'val', 'bad data type'
		assert num_threads > 0, 'lol no threads'
		self.params = params
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
		self.init_colors()
		self.start()

	def init_colors(self):
		self.colors = {}
		for color, id in enumerate(['white', 'brown', 'black', 'blue', 'red', 'green', 'pink', 'beige', 'clear', 'yellow',
									'orange', 'gray', 'purple', 'tan', 'silver', 'maroon', 'gold', 'blonde', 'sepia', 'plaid']):
			self.colors[color] = id
			self.colors[id] = color

	def start(self):
		self.process_list = []
		for i in range(self.num_threads):
			self.process_list.append(threading.Thread(target=self.next_batch_thread,
													  kwargs={'imgDirTrain': imgDirTrain, 'featDirTrain': featDirTrain}))

		for proc in self.process_list:
			proc.start()

	def kill(self):
		for proc in self.process_list:
			proc.join(timeout=0.1)

	def load_QA(self):
		annIds = self.vqa.getQuesIds(imgIds=[42, 74, 74, 133, 136, 139, 143, 164, 192, 196,
											 208, 241, 257, 283, 285, 294, 328, 338, 357, 359])
		# annIds = self.vqa.getQuesIds()
		if self.dataset_size is not None:
			annIds = annIds[:self.dataset_size]
		return self.vqa.loadQA(annIds)

	def index_to_color(self, id):
		return self.colors[id]

	def id_to_question(self, id=None):
		question = self.vqa.qqa[id]['question'][:-1].lower().split()
		if(self.max_ques_size < len(question)):
			raise Exception('Q too long')

		return [None] * (self.max_ques_size - len(question)) + list(map(lambda str: str.lower(), question))

	def id_to_answer(self, id=None):
		answer = self.vqa.loadQA(id)[0]['multiple_choice_answer'].lower()
		return answer

	def index_to_word(self, index):
		return self.word2vec.index_to_word(index)

	def visualize(self, ann, I):
		self.vqa.showQA([ann])
		plt.imshow(I)
		plt.axis('off')
		plt.show()

	def next_batch(self):
		return self.queue.get()

	def next_batch_thread(self, imgDirTrain, featDirTrain):
		while True:
			Anns, Is, Xs, Qs, As = {'b': [], 'n': [], 'm': [], 'c': []}, {'b': [], 'n': [], 'm': [], 'c': []}, {'b': [], 'n': [], 'm': [], 'c': []}, \
								   {'b': [], 'n': [], 'm': [], 'c': []}, {'b': [], 'n': [], 'm': [], 'c': []}
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
						I, X = scipy.misc.imread(imgDirTrain + imgFilename, mode='RGB'), np.load(featDirTrain + featFilename)
						I = scipy.misc.imresize(I, (224, 224, 3), 'bicubic') / 255.0
					elif (self.type == 'val'):
						I, X = scipy.misc.imread(imgDirVal + imgFilename, mode='RGB'), np.load(featDirVal + featFilename)
						I = scipy.misc.imresize(I, (224, 224, 3), 'bicubic') / 255.0

					Q = np.array([self.word2vec.word_to_index(word) for word in self.id_to_question(randomAnn['question_id'])])
					A = self.word2vec.word_to_index(self.id_to_answer(randomAnn['question_id']))
				except Exception as e:
					continue
				if randomAnn['answer_type'] == 'yes/no':
					type = 'b'
					A = 0 if self.id_to_answer(randomAnn['question_id']) == 'no' else 1
				elif randomAnn['answer_type'] == 'number':
					type = 'n'
					try:
						A = int(self.id_to_answer(randomAnn['question_id']))
						assert 0 <= A < self.params.num_range
					except:
						print('Number out of range!: ' + str(A))
						continue
				elif 'color' in randomAnn['question_type']:
					type = 'c'
					color = self.id_to_answer(randomAnn['question_id'])
					try:
						A = self.colors[color]
					except:
						print('Unknown color: ' + color)
						continue
				else:
					type = 'm'

				Anns[type].append(randomAnn)
				Is[type].append(I)
				Xs[type].append(X)
				Qs[type].append(Q)
				As[type].append(A)

			self.queue.put((np.array(Anns['b']), Is['b'], np.array(Xs['b']), np.array(Qs['b']), np.array(As['b']),
							np.array(Anns['n']), Is['n'], np.array(Xs['n']), np.array(Qs['n']), np.array(As['n']),
							np.array(Anns['m']), Is['m'], np.array(Xs['m']), np.array(Qs['m']), np.array(As['m']),
							np.array(Anns['c']), Is['c'], np.array(Xs['c']), np.array(Qs['c']), np.array(As['c'])))


class WordTable(object):
	def __init__(self):
		self.index_word()

	def index_word(self):
		self.word2idx = {'null': 0}
		self.idx2word = {0: 'null'}
		idx = 1
		for dataset in [VQA(annFileTrain, quesFileTrain), VQA(annFileVal, quesFileVal)]:
			for id, qqa in dataset.qqa.items():
				for word in [dataset.loadQA(id)[0]['multiple_choice_answer'].lower()] + qqa['question'][:-1].lower().split():
					if word in self.word2idx:
						continue
					self.word2idx[word] = idx
					self.idx2word[idx] = word
					idx += 1
		assert len(self.word2idx) == idx and len(self.idx2word) == idx
		self.vocab_size = idx
		pickle.dump(self, open('word2vec.cache', 'wb'))

	def word_to_index(self, word):
		if word not in self.word2idx:
			return 0
		return self.word2idx[word]

	def index_to_word(self, index):
		return self.idx2word[index]

	@staticmethod
	def load_word2vec():
		try:
			word2vec = pickle.load(open('word2vec.cache', 'rb'))
		except:
			word2vec = WordTable()
		return word2vec
