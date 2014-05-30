import random
import os
import cPickle
import numpy as np
from PIL import Image
import shutil
import re
import numpy as n

# Example on how to reload batch file and get back the actual images
#filename = '//msrne-vision/DNN/Batches-aquaticvertebrate/data_batch_0'
#fo = open(filename, 'rb')
#dict = cPickle.load(fo)
#x = dict['data']
#y = x[0:256**2*3,:]
#z = y[:,1]
#z = z.reshape(3,256,256)
#imr = Image.fromarray(z[0,:,:]);
#img = Image.fromarray(z[1,:,:]);
#imb = Image.fromarray(z[2,:,:]);
#img = Image.merge('RGB', (imr,img,imb))
#img.show();
#fo.close()


ROOT_IMAGE_DIR = "//msrne-vision/DNN/ImageNet/"
LABEL_DIR = "//msrne-vision/ImageNet/DNN/training/"

files = os.listdir(LABEL_DIR)

for f in range(1,len(files)):
	category = os.path.splitext(files[f])[0]
	ROOT_DIR = ROOT_IMAGE_DIR + category + "/"
	OUTPUT_DIR = "//msrne-vision/DNN/Batches-" + category + "/"
	if not os.path.exists(OUTPUT_DIR):
		os.makedirs(OUTPUT_DIR)
	DIM = 256
	images = []
	labels = []
	pattern = re.compile(r'.*.JPEG',re.I)
	labelNames = []

	term_list = open(LABEL_DIR + files[f], "r")
	label_index = 0
	for line in term_list:
		line_item = line.split(' ')
		label_name = line_item[0]
		label_name = label_name[1:]
		label_name = "n" + label_name.zfill(8)
		print "%d " % label_index
		if os.path.isdir(ROOT_DIR + label_name):
			imgs = os.listdir(ROOT_DIR + label_name)
			for img in imgs:
				if pattern.match(img):
					labels.append(label_index)
					images.append(ROOT_DIR + label_name + '/' + img)
		else:
			print("Can not found %s\n" % label_name)
		labelNames.append(label_index)
		label_index += 1

	print len(labels)
	print len(images)
	print len(labelNames)
	random_indexs = range(len(labels))
	random.shuffle(random_indexs)

	## meta data
	#for i in range(0,len(labels)):
	#        labelNames.append(labels[i])
	meta = {"label_names":labelNames,"num_vis":DIM*DIM*3,"num_cases_per_batch":1000,"data_mean":130.0*n.ones((DIM*DIM*3,1))}
	f=open(OUTPUT_DIR + r'batches.meta','wb')
	cPickle.dump(meta, f, protocol=cPickle.HIGHEST_PROTOCOL)
	f.close()

	BATCH_SIZE = 1000
	batch_indices = range(0, len(labels) / BATCH_SIZE)
	for batch_index in batch_indices:
		details = open(OUTPUT_DIR + "images_in_batch%d" % batch_index, "w")
		batch_labels = []
		batch_data = []
		for i in range(0,BATCH_SIZE):
			target_index = random_indexs[batch_index * BATCH_SIZE + i]
			batch_labels.append(labels[target_index])
			details.write(images[target_index] + "\t" + "%d" % labels[target_index] + "\n")
			image_data = Image.open(images[target_index])
			arr = np.array(image_data, order='C')
			im = np.fliplr(np.rot90(arr, k=3))
			batch_data.append(im.T.flatten('C'))
		details.close()
		flipDat = np.flipud(batch_data)
		rotDat = np.rot90(flipDat, k=3)
		batch = {'data':rotDat, 'labels':batch_labels}
		batch_stream=open(OUTPUT_DIR + "data_batch_%d" % batch_index,'wb')
		cPickle.dump(batch, batch_stream, True)
		batch_stream.close()
