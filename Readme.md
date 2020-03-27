This repository contains supporting code for the Computer Vision and Machine Learning analysis of ImageStream data collected for analysing the effect of age on Extracellular Vesicles. 

## Requirements
1. PIL >=1.1.7
2. skimage >=0.13.1
3. matplotlib >=2.1.1
4. pickle rev >=72223
5. scipy >=0.19.1
6. seaborn >=0.9.1
7. ipython and jupyter notebooks
8. tqdm

## Data pre-processing

To begin, store the data in a data folder somewhere on your computer accesible at `<path_to_data_folder>`. The data folder can contain multiple subfolders containing images.

In our case we had two subfolders called old and young. When you store the data, store the folder names in the format: `<folder_name>_np`.

The `<folder_name>` need to be populated in the `folders` list variable in the BioImage class. 

The images inside each folder are labeled in a specific format indexed preferably from 0 to N without leading zeros (in our case N=10000). 

The image naming format is: 
```
<folder_name>__<index>_Ch<channel_number>.ome.tif
```

The `<channel_number>` mappings should also be populated in the channel variable. Note that if this is changed in the codebase, then there are going to be lots of micro edits required to move the channels around. When this software was written, we did not expect switching channels in and out. 


This format can be changed in the `__init__` function of the `BioImage` class.

### Folders automatically created:
1. Pickle folders: For every `<folder_name>_np` in your data repository, the code creates an empty `<folder_name>_pickle` folder. This will contain optimized, combined versions of the tif files that can be loaded much quicker than the tiff files. 
2. Data folder: Contains the features computed using the `BioAnalysis` class as `.npy` files. 

### Pre-processing the tif files

The format exported by imagestream contains non-standard elements and therefore cannot be opened by python's image processing libraries. An easy hack to get around this problem is to run a batch convert operation from tif to tif using the ImageJ application. 

## Computing Data variables using the `BioAnalysis` class

In a python window, either using a python ide or jupyter notebooks, import the Bioanalysis class and create an object as shown below. 

```
import BioAnalysis
analysis = BioAnalysis('<path_to_data_folder>')
```

This will start a series of computations that will loop through each image in the data folder. Once all these are complete, you can start playing around with the generated data. For a reference on how to do this, inspect the ipynb files. 

