import numpy as np
from matplotlib import pyplot as plt
from BioImage import BioImage
from scipy.cluster.vq import vq, kmeans, whiten
from random import randint

data = np.load('dataset.npy')
folders = ['Y_2_converted', 'Y_3_converted', 'Y_4_converted', 'KO_1_converted', 'KO_2_converted', 'O_1_converted', 'O_2_converted']


# Calculate sphericity
amplitude1 = data[:,3]
height1 = data[:,2]
width1= np.abs(data[:,6:8])
amplitude2 = data[:,10]
height2 = np.expand_dims(data[:,9], axis=1)
width2= np.abs(data[:,13:15])

circularity1_up = (np.amax(width2, axis=1))
circularity1_down = (np.amin(width2, axis=1))
circularity = np.divide(circularity1_down,circularity1_up)

idx = randint(0,70000)
print(circularity1_up[idx])
print(circularity1_down[idx])
print(circularity[idx])

b = BioImage(folders[int(data[idx,0])], int(data[idx,1]))
# b.showImage()
circularity = np.expand_dims(circularity, axis=1)
amplitude2 = np.expand_dims(amplitude2, axis=1)
clustering_dataset = np.concatenate((circularity, amplitude2, width2), axis=1)

# filtered = whiten(data[:,[6,7,13,14]])
filtered = whiten(clustering_dataset)
# filtered = data[:,2:]
num_clusters = 30
centroids,_ = kmeans(filtered, num_clusters)
print(centroids.shape)

idx,_ = vq(filtered, centroids)

cluster9 = data[idx==8,:]
cluster19 = data[idx==18,:]
for i in range(0, num_clusters):
    print (data[idx==i, :].shape)
counts = np.bincount(idx)
cluster_number = np.argmax(counts)

largest_cluster = data[idx==cluster_number,:]
np.random.shuffle(largest_cluster)
for folder, cluster in zip(largest_cluster[:,0], largest_cluster[:,1]):
    b = BioImage(folders[int(folder)], int(cluster))
    b.showImage()
    plt.waitforbuttonpress()
    plt.close()


