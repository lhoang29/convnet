import sys
import numpy as np
import cPickle as pickle
import os

os.environ['PYTHON_EGG_CACHE'] = './'  
import struct
import time
from convdata import *

infile = sys.argv[1]
outfile = sys.argv[2]

dic = unpickle(infile)

f = open(outfile, "wb")
f.write(struct.pack('i', len(dic['data'])))
f.write(struct.pack('i', len(dic['data'][0])))

for i in range(0, len(dic['data'])):
	for j in range(0, len(dic['data'][i])):
		f.write(struct.pack('f', dic['data'][i][j]))
f.close()
