from PIL import Image
import skimage
from skimage import data, color, img_as_ubyte
from matplotlib import pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from random import randint
import gaussfitter
from scipy import ndimage
from scipy import ndimage as ndi
from skimage.morphology import extrema
from skimage import exposure
from skimage.measure import label
from skimage.morphology import reconstruction, binary_closing
from scipy.ndimage.morphology import binary_fill_holes
from skimage.filters import threshold_isodata
from skimage.measure import perimeter
import os

#folders = ['Y_2_converted', 'Y_3_converted', 'Y_4_converted','O_1_converted', 'O_2_converted']
folders = ['old', 'young']
channel = {1:'B/W', 2:'CD63', 3: 'PanEV', 4: 'alt KL', 6: 'SSC', 7:'cd81', 9: 'B/W', 11:'WT KL'}

class BioImage:
  
    def __init__(self,  folder, num, root = '../Amrita_29feb'):
        base_folder = root
        pickle_folder = base_folder + folder + '_pickle/'
        header_pickle= pickle_folder + folder + '__' + str(num) + '.p'
        if not os.path.exists(pickle_folder):
            os.mkdir(pickle_folder)
        self.name = folder + ' : ' + str(num)
        self.mask = None
        try:
            self.image = pickle.load(open(header_pickle, 'rb'))
            self.ch1 = self.image['Ch1 - B/W']
            self.ch2 = self.image['Ch2 - CD63']
            self.ch3 = self.image['Ch3 - PanEV']
            try:
                self.ch4 = self.image['Ch4 - ']
            except:
                pass
            self.ch6 = self.image['Ch6 - Side Scatter']
            self.ch7 = self.image['Ch7 - CD81']
            self.ch9 = self.image['Ch9 - B/W']
            self.ch11 = self.image['Ch11 - CD9']
            
        except:
            header = base_folder + folder + '_np/' + folder + '__' + str(num) + '_'
            self.ch1 = Image.open(header + 'Ch1.ome.tif')
            self.ch2 = Image.open(header + 'Ch2.ome.tif')
            self.ch3 = Image.open(header + 'Ch3.ome.tif')
            try:
                self.ch4 = Image.open(header + 'Ch4.ome.tif')
            except:
                pass
            self.ch6 = Image.open(header + 'Ch6.ome.tif')
            self.ch7 = Image.open(header + 'Ch7.ome.tif')
            self.ch9 = Image.open(header + 'Ch9.ome.tif')
            self.ch11 = Image.open(header +'Ch11.ome.tif')
            # Convert to nparray and normalize to 4000:
            self.ch1 = np.array(self.ch1)/4000.0
            self.ch2 = np.array(self.ch2)/4000.0
            self.ch3 = np.array(self.ch3)/4000.0
            try: 
                self.ch4 = np.array(self.ch4)/4000.0
            except:
                pass
            self.ch6 = np.array(self.ch6)/4000.0
            self.ch7 = np.array(self.ch7)/4000.0
            self.ch9 = np.array(self.ch9)/4000.0
            self.ch11 = np.array(self.ch11)/4000.0
            try:
                self.image = {'Ch' + str(1 )+ channel[1]:self.ch1, 
                              'Ch' + str(9 )+ channel[9 ]:self.ch9, 
                              'Ch' + str(6 )+ channel[6 ]:self.ch6, 
                              'Ch' + str(2 )+ channel[2 ]:self.ch2, 
                              'Ch' + str(3 )+ channel[3 ]:self.ch3, 
                              'Ch' + str(4 )+ channel[4 ]:self.ch4, 
                              'Ch' + str(7 )+ channel[7 ]:self.ch7, 
                              'Ch' + str(11)+ channel[11]:self.ch11}
            except:
                self.image = {'Ch' + str(1 )+ channel[1]:self.ch1, 
                              'Ch' + str(9 )+ channel[9 ]:self.ch9, 
                              'Ch' + str(6 )+ channel[6 ]:self.ch6, 
                              'Ch' + str(2 )+ channel[2 ]:self.ch2, 
                              'Ch' + str(3 )+ channel[3 ]:self.ch3, 
                              'Ch' + str(7 )+ channel[7 ]:self.ch7, 
                              'Ch' + str(11)+ channel[11]:self.ch11}
            pickle.dump(self.image, open(header_pickle,'wb'))


    def showImage(self):
        plt.figure()
        plt.suptitle(self.name)
        i = 1
        for ch_name, image in self.image.items():
            plt.subplot(2,4,i)
            i = i + 1
            plt.imshow(image, cmap='magma')
            plt.title(ch_name)
            plt.axis('off')
        plt.show()

    def showPipeline(self):
        self.showImage();
        self.show3D(self.ch1)
        self.show3D(self.ch6)
        self.showMaxima(self.ch1)
        self.showMaxima(self.ch6)
        return
        

    def show3D(self, im):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        xx, yy = np.mgrid[0:im.shape[0], 0:im.shape[1]]
        ax.plot_surface(xx, yy, im, rstride=1, cstride=1, cmap='magma',  linewidth=0)
        plt.axis('on')
        plt.tight_layout()
        plt.show()
        return

    def showMaxima(self, im):
        fig = plt.figure()
        plt.imshow(im)
        im = self.runHistogramAbs(im)
        im = ndimage.gaussian_filter(im, sigma=1)
        im = exposure.rescale_intensity(im)
        h = 0.2
        h_maxima = extrema.h_maxima(im, h)
        label_h_maxima = label(h_maxima)
        img = skimage.exposure.rescale_intensity(im)
        overlay_h = color.label2rgb(label_h_maxima, img, alpha=0.7,
                            bg_label=0,
                            bg_color=None)
        plt.imshow(overlay_h)

        plt.show()
        return
        

        
    def getExtrema(self):
        Extrema = np.array([0,0,0])
        
        img = self.ch1
        image_gray = img
        image_rgb = img
        im = self.runHistogramAbs(img)
        # im = ndimage.gaussian_filter(im, sigma=0.7)
        # im = exposure.rescale_intensity(im)
        im = ndimage.gaussian_filter(im, sigma=1)
        im = exposure.rescale_intensity(im)
        h = 0.2
        h_maxima = extrema.h_maxima(im, h)
        label_h_maxima = label(h_maxima)
        Extrema[0] = np.count_nonzero(label_h_maxima)
               
        #check if the extrema is right next to the edge
        if Extrema[0]==1:
            indices = np.nonzero(h_maxima)
            distance = im.shape[1] - indices[1]
            Extrema[2] = distance[0]
        
        
        img = self.ch6
        image_gray = img
        image_rgb = img
        im = self.runHistogramAbs(img)
        # im = ndimage.gaussian_filter(im, sigma=0.7)
        im = ndimage.gaussian_filter(im, sigma=1)
        im = exposure.rescale_intensity(im)
        h = 0.2
        h_maxima = extrema.h_maxima(im, h)
        label_h_maxima = label(h_maxima)
        Extrema[1] = np.count_nonzero(label_h_maxima)
        params = self.getParams()
        Extrema = np.append(Extrema, params)
        return Extrema
        
    def getFeatures(self):
        features = []
        # Preprocess the ch1 image -- let's just try an abs 
        img = self.ch1
        hist,bins = np.histogram(img.ravel(),256,[0,np.max(img)])
        #plt.hist(img.ravel(),256,[0,np.max(img)]); plt.show()
        zero = np.argmax(hist)
        img = img - bins[zero]
        ch1 = np.abs(img)
        ch1 = ndimage.gaussian_filter(ch1, sigma=2)
        ch6 = self.ch6
        ch6 = ndimage.gaussian_filter(ch6, sigma=2)
        features = np.concatenate([features, gaussfitter.gaussfit(ch1)])
        features = np.concatenate([features, gaussfitter.gaussfit(ch6)])
        return features

    def runHistogramAbs(self, img):
        hist,bins = np.histogram(img.ravel(),256,[0,np.max(img)])
        zero = np.argmax(hist)
        img = img - bins[zero]
        im = np.abs(img)
        # im[im<0] = 0
        # im = im/np.max(im)
        return im
    def runHistogram(self, img):
        hist,bins = np.histogram(img.ravel(),256,[0,np.max(img)])
        zero = np.argmax(hist)
        img = img - bins[zero]
        # im = np.abs(img)
        img[img<0] = 0
        # im = img/np.max(img)
        return img

    def getThresholded(self, img, abs=True):
        if abs:
            im = self.runHistogramAbs(img)
        else:
            im =self.runHistogram(img)
        try:
            threshold = threshold_isodata(im)
        except:
            return img
        im = im > threshold
        filled = binary_fill_holes(im).astype(int)
        mask = binary_closing(filled)
        return mask

    def getCh2Activation(self):
        ch1 = self.ch1
        ch1 = self.getThresholded(ch1)
        
        ch2 = self.ch2
        ch2 = self.getThresholded(ch2,False)

        img = np.multiply(ch1,ch2)
        return np.count_nonzero(img)>0

    def getCh4Activation(self):
        ch1 = self.ch1
        ch1 = self.getThresholded(ch1)

        ch4 = self.ch4
        h = 0.08
        h_maxima = extrema.h_maxima(ch4, h)
        label_h_maxima = np.multiply(ch1, label(h_maxima))
        return np.count_nonzero(label_h_maxima)>0

    def getCh7Activation(self):
        ch1 = self.ch1
        ch1 = self.getThresholded(ch1)
        ch2 = self.ch7
        ch2 = self.runHistogram(ch2)
        ch2 = ndi.gaussian_filter(ch2, sigma=3)
        h = 0.08
        h_maxima = extrema.h_maxima(ch2, h)
        label_h_maxima = np.multiply(ch1, label(h_maxima))
        return np.count_nonzero(label_h_maxima)>0
    def getCh11Activation(self):
        ch1 = self.ch1
        ch1 = self.getThresholded(ch1)
        ch2 = self.ch11
        ch2 = self.runHistogram(ch2)
        ch2 = ndi.gaussian_filter(ch2, sigma=3)
        h = 0.08
        h_maxima = extrema.h_maxima(ch2, h)
        label_h_maxima = np.multiply(ch1, label(h_maxima))
        return np.count_nonzero(label_h_maxima)>0
    
    def getActivations(self):
        activations  = np.array([False,False,False])

        activations[0] = self.getCh2Activation()
        activations[1] = self.getCh7Activation()
        activations[2] = self.getCh11Activation()
        return activations

    def getActivationSize(self, ch):
        ch = self.getThresholded(ch, False)
        return np.count_nonzero(ch)

    def getActivationSizes(self):
        return np.array([self.getActivationSize(self.ch2),
                         self.getActivationSize(self.ch7),
                         self.getActivationSize(self.ch11)])


    
    def getParams(self):
        img = self.ch1
        hist,bins = np.histogram(img.ravel(),256,[0,np.max(img)])
        #plt.hist(img.ravel(),256,[0,np.max(img)]); plt.show()
        zero = np.argmax(hist)
        img = img - bins[zero]
        im = np.abs(img)
        threshold = threshold_isodata(im)
        im = im > threshold
    
        filled = binary_fill_holes(im).astype(int)
        filled = binary_closing(filled)

        A1 = np.count_nonzero(filled)
        P1 = perimeter(filled, neighbourhood=16)
        C1 = 4*3.14*A1/(P1*P1)
        
        img = self.ch6
        hist,bins = np.histogram(img.ravel(),256,[0,np.max(img)])
        #plt.hist(img.ravel(),256,[0,np.max(img)]); plt.show()
        zero = np.argmax(hist)
        img = img - bins[zero]
        im = np.abs(img)
        threshold = threshold_isodata(im)
        im = im > threshold
    
        filled = binary_fill_holes(im).astype(int)
        filled = binary_closing(filled)

        A2 = np.count_nonzero(filled)
        P2 = perimeter(filled, neighbourhood=16)
        C2 = 4*3.14*A2/(P2*P2)
        params = np.array([A1,P1,C1,A2,P2,C2])
        return params
        
    

