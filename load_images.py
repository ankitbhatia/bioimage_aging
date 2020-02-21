#!/usr/local/bin/python3
from random import shuffle, randint
from PIL import Image
import glob
from BioImage import BioImage

shuffle_data = True  # shuffle the addresses before saving
hdf5_path = '../dataset.hdf5'  # address to where you want to save the hdf5 file
data_path = '../*/*'

# read addresses and labels from the 'train' folder. 
addrs = glob.glob(data_path)
labels = [0 if 'Y' in addr else 1 for addr in addrs]  # 0 = YOUNG, 1 = OLD

#
# to shuffle data
if shuffle_data:
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)

# for i in range(0,10):
#     print(str(labels[i]) + " file: " + str(addrs[i]))
# # Divide the hata into 60% train, 20% validation, and 20% test
# train_addrs = addrs[0:int(0.6*len(addrs))]
# train_labels = labels[0:int(0.6*len(labels))]
#
# val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
# val_labels = labels[int(0.6*len(addrs)):int(0.8*len(addrs))]
#
# test_addrs = addrs[int(0.8*len(addrs)):]
# test_labels = labels[int(0.8*len(labels)):]

label = ['Ch1', 'Ch2', 'Ch4', 'Ch6', 'Ch7', 'Ch11']
folders = ['Y_2_converted', 'Y_3_converted', 'Y_4_converted', 'KO_1_converted', 'KO_2_converted', 'O_1_converted', 'O_2_converted']

num = randint(0,10000)
folder = folders[randint(0,len(folders)-1)]
b = BioImage(folder, num)
b.showImage()
b.show3D(5)
print(b.getFeatures())

