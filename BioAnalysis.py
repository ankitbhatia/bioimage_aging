from BioImage import BioImage, folders, channel
from scipy.stats import fisher_exact
import numpy as np
from tqdm import tqdm, trange
from numpy.random import randint
import sys
from IPython.display import HTML, display
import tabulate
import matplotlib.pyplot as plt
import os



class BioAnalysis:
    def __init__(self, root = None):

        self.root = root
        self.data_path = self.root + '/Data/'
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)

        try:
            self.dataset = np.load(self.data_path + 'dataset.npy')
        except:
            print('Could not load dataset, computing features')
            sys.stdout.flush()
            self.dataset = None
            self.computeFeatures()
            self.saveDataset()
        try:
            self.filtered_dataset = np.load(self.data_path + 'filtered_dataset.npy')
        except:
            print('Could not load filtered dataset, computing filters')
            sys.stdout.flush()
            self.filtered_dataset = None
            self.runFilter()
            self.saveFilteredDataset()
        try:
            self.activations = np.load(self.data_path + 'activations.npy')
        except:
            print('Could not load activations, computing activations')
            sys.stdout.flush()
            self.activations = None
            self.computeActivations()
            self.saveActivations()
        try:
            self.points_young = np.load(self.data_path + 'points_young.npy')
            self.points_old = np.load(self.data_path + 'points_old.npy')
            self.points = np.load(self.data_path + 'points.npy')
        except:
            print('Could not load points, computing points')
            sys.stdout.flush()
            self.points_young= None
            self.points_old = None
            self.computePoints()
            self.savePoints()
        return

    def getFilteredIndices(self):
        filtered = self.filtered_dataset
        return filtered[:,0:2]



    def computeFeatures(self):
        self.dataset = np.array([])
        for folder_idx, folder in enumerate(folders):
            print('Processing folder ' + folder + ':')
            sys.stdout.flush()
            for i in trange(0,10000):
                b = BioImage(folder, i, self.root)
                try: 
                    features = b.getExtrema()
                except Exception as e: 
                    print(e)
                    b.showImage()
                data_line = np.insert(features, 0, [folder_idx, i])
                if self.dataset.size == 0:
                    self.dataset = data_line
                else:
                    self.dataset = np.vstack((self.dataset, data_line))

        return
    def runFilter(self):
        self.filtered_dataset = self.dataset
        threshold = 5
        # new_data[2 & 3] = number of maxima in ch1 and ch6
        # new_data[4] = distance from edge
        # new_Data[7] circularity in ch1
        self.filtered_dataset = self.filtered_dataset[(self.filtered_dataset[:,2]<=1) 
                    & (self.filtered_dataset[:,3]<=1)] 
                    #& (self.filtered_dataset[:,4]>threshold)
                    #& (self.filtered_dataset[:,7]>0.5)
                    #& (self.filtered_dataset[:,7]<1.25)]
        self.filtered_size = self.filtered_dataset.shape[0]
        return

    def showImage(self, folder, sample):
        bioimage = BioImage(folder, sample, self.root)
        bioimage.showImage()
        features = bioimage.getExtrema()
        print('Number of ch1 maxima: ', features[0])
        print('Number of ch6 maxima: ', features[1])
        print('Distance from edge: ', features[3])
        print('Circularity Ch1: ', features[5])
        print('Circularity Ch6: ', features[8])
        return bioimage


    def showRandomImage(self):
        index = randint(0, self.dataset.shape[0])
        folder = folders[int(self.dataset[index,0])]
        return self.showImage(folder, int(self.dataset[index, 1]))
    
    def showRandomFilteredImage(self):
        index = randint(0, self.filtered_dataset.shape[0])
        folder = folders[int(self.filtered_dataset[index, 0])]
        return self.showImage(folder, int(self.filtered_dataset[index, 1]))


    def runPrefilter(self):
        if self.dataset is None: 
            self.computeFeatures()
        self.runFilter()
        return

    def computeActivations(self):
        self.activations = np.array([])
        row_one = True
        for row in tqdm(self.filtered_dataset):
            folder = folders[int(row[0])]
            index = int((row[1]))
            b = BioImage(folder, index, self.root)
            if row_one:
                #print(b.getActivations())
                self.activations = b.getActivations()
                row_one = False
            else:
                self.activations = np.vstack((self.activations, b.getActivations()))

    def getContingencyMatrix(self):

        data_activations = np.hstack((self.filtered_dataset,self.activations))
        young = data_activations[(data_activations[:,0]==1)]
        old = data_activations[(data_activations[:,0]==0)]
        print('Number of Young samples:', young.shape[0])
        print('Number of Old Samples:', old.shape[0])
        young_counts = np.sum(young[:,11:14], axis=0)
        old_counts = np.sum(old[:,11:14], axis=0)
        young_none = np.count_nonzero(~np.any(young[:,11:14], axis=1))
        old_none = np.count_nonzero(~np.any(old[:,11:14], axis=1))
        counts = np.vstack((young_counts, old_counts))
        np.unique(self.filtered_dataset[:,0],return_counts=True)
        young_total = young.shape[0]
        old_total = old.shape[0]
        contingency_ch2 = np.zeros((2,2))
        contingency_ch2[0,0] = young_counts[0]
        contingency_ch2[0,1] = old_counts[0]
        contingency_ch2[1,0] = young_total - young_counts[0]
        contingency_ch2[1,1] = old_total - old_counts[0]
        self.printContingencyMatrix(contingency_ch2, ["CD63+", 'CD63-'], ['Young', 'Old'], fisher_exact(contingency_ch2)[1])
        contingency_ch7 = np.zeros((2,2))
        contingency_ch7[0,0] = young_counts[1]
        contingency_ch7[0,1] = old_counts[1]
        contingency_ch7[1,0] = young_total - young_counts[1]
        contingency_ch7[1,1] = old_total - old_counts[1]
        self.printContingencyMatrix(contingency_ch7, ["CD81+", 'CD81-'], ['Young', 'Old'], fisher_exact(contingency_ch7)[1])
        contingency_ch11 = np.zeros((2,2))
        contingency_ch11[0,0] = young_counts[2]
        contingency_ch11[0,1] = old_counts[2]
        contingency_ch11[1,0] = young_total - young_counts[2]
        contingency_ch11[1,1] = old_total - old_counts[2]
        self.printContingencyMatrix(contingency_ch11, ["CD9+", 'CD9-'], ['Young', 'Old'], fisher_exact(contingency_ch11)[1])
        return

    def computePoints(self):
        if self.points_young is None:
            rowOld = True
            rowYoung = True
            rowPoints = True
            points = np.array([])
            for row in tqdm(self.filtered_dataset):
                folder = folders[int(row[0])]
                index = int((row[1]))
                b = BioImage(folder, index, self.root)
                # channel 2
                mask = b.getThresholded(b.ch2)
                ch2 = np.sum(np.multiply(mask, b.ch2))
                # Channel 7
                mask = b.getThresholded(b.ch7)
                ch7 = np.sum(np.multiply(mask, b.ch7))
                #channel 11:
                mask = b.getThresholded(b.ch11)
                ch11 = np.sum(np.multiply(mask, b.ch11))
                try:
                    # channel 4
                    mask = b.getThresholded(b.ch4)
                    ch4 = np.sum(np.multiply(mask, b.ch4))
                    p = np.array([ch2, ch7, ch11,ch4])
                except:
                    p = np.array([ch2, ch7, ch11])

                s = b.getActivationSizes()
                p = np.append(p,s)
                if rowPoints:
                    points = p
                    rowPoints = False
                else:
                    points = np.vstack((points, p))


                if row[0]==0: # Old
                    if rowOld: 
                        points_old = p
                        rowOld = False
                    else:
                        points_old = np.vstack((points_old, p))
                else: #Young
                    if rowYoung:
                        points_young = p
                        rowYoung = False
                    else:
                        points_young = np.vstack((points_young,p))
            self.points_young = points_young
            self.points_old= points_old
            self.points = points
        return

    def getTotalIntensityDistribution(self, plots_on = True, hist_range = []):
        points_old = self.points_old
        points_young = self.points_young
        # compute means and variance of old and young
        mean_old = np.mean(points_old, axis=0)
        var_old = np.var(points_old, axis=0)
        mean_young = np.mean(points_young, axis=0)
        var_young = np.var(points_young, axis=0)
        if plots_on:
            if not hist_range:
                self.plotIntensityDistribution("Intensity", points_young[:,0:3], points_old[:,0:3])
                self.plotSizeDistribution("Size", points_young[:,3:6], points_old[:,3:6])
            else:
                self.plotIntensityDistribution("Intensity", points_young[:,0:3], points_old[:,0:3], hist_range)
                self.plotSizeDistribution("Size", points_young[:,3:6], points_old[:,3:6],hist_range)
        # tests
        return {'Old':{'Mean':mean_old, 'Var': var_old}, 'Young':{'Mean':mean_young, 'Var': var_young}}

    def plotIntensityDistribution(self, title, points_young, points_old, hist_range=[]):
        plt.figure()
        ax1 = plt.subplot(3, 1, 1)
        plt.hist(points_young[:, 0], bins = 1000, density=True, color='r', histtype='step')
        plt.hist(points_old[:, 0], bins = 1000, density = True, color='b', histtype='step')
        plt.legend(['young', 'old'])
        plt.title(title + ' ' + channel[2])
        ax2 = plt.subplot(3, 1, 2)
        plt.hist(points_young[:, 1], bins = 1000, density=True, color='r', histtype='step')
        plt.hist(points_old[:, 1], bins = 1000, density = True, color='b', histtype='step')
        plt.legend(['young', 'old'])
        plt.title(channel[7])
        ax3= plt.subplot(3,1,3)
        plt.hist(points_young[:, 2], bins = 1000, density=True, color='r', histtype='step')
        plt.hist(points_old[:, 2], bins = 1000, density = True, color='b', histtype='step')
        plt.legend(['young', 'old'])
        plt.title(channel[11])
        plt.tight_layout()
        if hist_range:
            ax1.set_xlim(hist_range[0])
            ax2.set_xlim(hist_range[1])
            ax3.set_xlim(hist_range[2])
        return
    def plotSizeDistribution(self, title, points_young, points_old, hist_range=[]):
        plt.figure()
        ax1 = plt.subplot(3, 1, 1)
        plt.hist(points_young[:, 0], bins = 100, density=True, color='r', histtype='step')
        plt.hist(points_old[:, 0], bins = 100, density = True, color='b', histtype='step')
        plt.legend(['young', 'old'])
        plt.title(title + ' ' + channel[2])
        ax2 = plt.subplot(3, 1, 2)
        plt.hist(points_young[:, 1], bins = 100, density=True, color='r', histtype='step')
        plt.hist(points_old[:, 1], bins = 100, density = True, color='b', histtype='step')
        plt.legend(['young', 'old'])
        plt.title(channel[7])
        ax3 = plt.subplot(3, 1, 3)
        plt.hist(points_young[:, 2], bins = 100, density=True, color='r', histtype='step')
        plt.hist(points_old[:, 2], bins = 100, density = True, color='b', histtype='step')
        plt.legend(['young', 'old'])
        plt.title(channel[11])
        plt.tight_layout()
        if hist_range:
            ax1.set_xlim(hist_range[3])
            ax2.set_xlim(hist_range[4])
            ax3.set_xlim(hist_range[5])

        return


        

    def printContingencyMatrix(self, matrix, row_header, column_header, p_value):
        table = [['',column_header[0],column_header[1]],
                 [row_header[0],matrix[0,0], matrix[0,1]],
                 [row_header[1],matrix[1,0],matrix[1,1]]]
        display(HTML(tabulate.tabulate(table, tablefmt='html')))
        print('p-value:', p_value)
        return




    def saveDataset(self):
        if self.dataset is not None:
            np.save(self.data_path + 'dataset', self.dataset)
    def saveFilteredDataset(self):
        if self.filtered_dataset is not None:
            np.save(self.data_path + 'filtered_dataset', self.filtered_dataset)
    def saveActivations(self):
        if self.activations is not None:
            np.save(self.data_path + 'activations', self.activations)
    def savePoints(self):
        if self.points_young is not None:
            np.save(self.data_path + 'points_young', self.points_young)
        if self.points_old is not None:
            np.save(self.data_path + 'points_old', self.points_old)
        if self.points is not None:
            np.save(self.data_path + 'points', self.points)
        





