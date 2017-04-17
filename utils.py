import pickle
import threading, re
from queue import Queue
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from VQA.PythonHelperTools.vqaTools.vqa import VQA

dataDir = '/home/victor/VQA'
taskType = 'OpenEnded'
dataType = 'mscoco'
dataSubTypeTrain = 'train2014'
AnnoSubTypeTrain = 'train2017'
annFileTrain = '%s/Annotations/%s_%s_annotations.json' % (dataDir, dataType, AnnoSubTypeTrain)
quesFileTrain = '%s/Questions/%s_%s_%s_questions.json' % (dataDir, taskType, dataType, AnnoSubTypeTrain)
imgDirTrain = '%s/Images/%s/%s/' % (dataDir, dataType, dataSubTypeTrain)
featDirTrain = '%s/Features/%s/%s/' % (dataDir, dataType, dataSubTypeTrain)

dataSubTypeVal = 'val2014'
AnnoSubTypeVal = 'val2017'
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
		self.start()

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
		# annIds = self.vqa.getQuesIds(imgIds=[42, 74, 74, 133, 136, 139, 143, 164, 192, 196])
		annIds = self.vqa.getQuesIds()
		# if self.dataset_size is not None:
		# 	annIds = annIds[:self.dataset_size]
		return self.vqa.loadQA(annIds)

	# def id_to_question(self, id=None):
	# 	question = self.vqa.qqa[id]['question'][:-1].split()
	# 	return [None] * (self.max_ques_size - len(question)) + list(map(lambda str: str.lower(), question))
	#
	# def id_to_answer(self, id=None):
	# 	ans_dict = defaultdict(lambda: 0)
	# 	for answer in self.vqa.loadQA(id)[0]['answers']:
	# 		if len(answer['answer'].split()) == 1:
	# 			ans_dict[answer['answer']] += 1
	# 	return max(ans_dict, key=lambda k: ans_dict[k])

	def id_to_question(self, id=None):
		question = self.vqa.qqa[id]['question'][:-1].lower().split()
		Q_strip_apostrophe = []
		for word in question:
			if word is None:
				Q_strip_apostrophe.append(word)
			else:
				for a in re.split(r"['/\\?!,-.\"]", word):
					if a is not '':
						Q_strip_apostrophe.append(a)

		if(self.max_ques_size < len(Q_strip_apostrophe)):
			raise Exception('Q too long')

		return [None] * (self.max_ques_size - len(Q_strip_apostrophe)) + list(
			map(lambda str: str.lower(), Q_strip_apostrophe))

	def id_to_answer(self, id=None):
		#ans_dict = defaultdict(lambda: 0)
		answer = re.split(r"[' /\\?!,-.\"]", self.vqa.loadQA(id)[0]['multiple_choice_answer'])
		if len(answer) == 1:
			return answer[0]
		else:
			raise Exception('A too long')

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
						I, X = io.imread(imgDirTrain + imgFilename), np.load(featDirTrain + featFilename)
					elif (self.type == 'val'):
						I, X = io.imread(imgDirVal + imgFilename), np.load(featDirVal + featFilename)

					Q = np.stack([self.word2vec.word_vector(word) for word in self.id_to_question(randomAnn['question_id'])])
					A = self.word2vec.word_to_index(self.id_to_answer(randomAnn['question_id']))
				except Exception as e:
					print("bad !" + str(e) + ", Orig Ques: " + str(self.vqa.qqa[randomAnn['question_id']]['question'][:-1].lower().split()) + ", Orig Answer: " + str(self.vqa.loadQA(randomAnn['question_id'])[0]['multiple_choice_answer']))
					#print(self.vqa.loadQA(randomAnn['question_id'])[0]['multip'])
					continue
				if randomAnn['answer_type'] == 'yes/no':
					type = 'b'
					if self.id_to_answer(randomAnn['question_id']) == 'no':
						A = 0
					else:
						A = 1
					#A = np.zeros(2, dtype=np.float32)
					#A[ans] = 1
				elif randomAnn['answer_type'] == 'number':
					type = 'n'
					try:
						A = int(self.id_to_answer(randomAnn['question_id']))
						assert 0 <= A < self.params.num_range
						#A = np.zeros(self.params.num_range, dtype=np.float32)
						#A[ans] = 1
					except:
						print('bad number oo range!: ' + str(A))
						continue
				elif 'color' in randomAnn['question_type']:
					type = 'c'
					colors = {
						'white': 0,
						'brown': 1,
						'black': 2,
						'blue': 3,
						'red': 4,
						'green': 5,
						'pink': 6,
						'beige': 7,
						'clear': 8,
						'yellow': 9,
						'orange': 10,
						'gray': 11,
						'purple': 12,
						'tan': 13,
						'silver': 14,
						'maroon': 15,
						'gold': 16,
						'blonde': 17,
						'sepia': 18,
						'plaid': 19,
					}
					try:
						A = colors[self.id_to_answer(randomAnn['question_id'])]
					except:
						print('Unknown color: ' + str(self.id_to_answer(randomAnn['question_id'])))
						continue
				else:
					type = 'm'
					#ans = self.word2vec.word_to_index(self.id_to_answer(randomAnn['question_id']))
					#A = np.zeros(self.params.vocab_size, dtype=np.float32)
					#A[ans] = 1

				Anns[type].append(randomAnn)
				Is[type].append(I)
				Xs[type].append(X)
				Qs[type].append(Q)
				As[type].append(A)

				#print(type + ", Proc'd Ques: " + str(self.id_to_question(randomAnn['question_id'])) + ", Proc'd Answer: " + str(self.id_to_answer(randomAnn['question_id'])))



			#print("m's: " + str(len(Qs['m'])) + ", n's: " + str(len(Qs['n'])) + ", b's: " + str(len(Qs['b'])))
			#print(As['m'])
			#print(Qs['m'])
			self.queue.put((np.array(Anns['b']), Is['b'], np.array(Xs['b']), np.array(Qs['b']), np.array(As['b']),
							np.array(Anns['n']), Is['n'], np.array(Xs['n']), np.array(Qs['n']), np.array(As['n']),
							np.array(Anns['m']), Is['m'], np.array(Xs['m']), np.array(Qs['m']), np.array(As['m']),
							np.array(Anns['c']), Is['c'], np.array(Xs['c']), np.array(Qs['c']), np.array(As['c'])))


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
	# def one_hot(self, word):
	# 	return self.word2idx[word]

	def word_to_index(self, word):
		return self.word2idx[word]

	def index_to_word(self, index):
		return self.idx2word[index]

	def load_glove(self, dim):
		word2vec = {}

		path = 'VQA/PythonHelperTools/glove.6B/glove.6B.' + str(dim) + 'd'
		try:
			with open(path + '.cache', 'rb') as cache_file:
				word2vec = pickle.load(cache_file)

		except:
			with open(path + '.txt') as f:
				for line in f:
					l = line.rstrip().split()
					word2vec[l[0]] = [float(x) for x in l[1:]]

			with open(path + '.cache', 'wb') as cache_file:
				pickle.dump(word2vec, cache_file)

		print("Glove data loaded")
		return word2vec
