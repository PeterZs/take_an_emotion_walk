# taew

Our scripts have been tested on Ubuntu 18.04 LTS with
- Python 3.6
- Cuda 10.2
- cudNN 7.6.5

## Installing Requirements

We recommend using an Anaconda virtual environment. If Anaconda is not already installed, [Install Anaconda](https://www.anaconda.com/distribution/#download-section) and run
``` 
conda env create -n taew -f environment.yml
```
from within the project directory

## Download datasets and network weights

Run the following command from within the project directory to download and extract the sample datasets and network weights:
``` 
sh download_data_weights.sh
```

## Evaluation

1. Activate the conda environment
```
conda activate taew
```
2. Run the evaluation script
For dgnn evaluation:
```
python evaluate.py --dgnn 
```
For stgcn evaluation:
```
python evaluate.py --stgcn
```
For lstm network evaluation:
```
python evaluate.py --lstm
```
For step evaluation:
```
python evaluate.py --step
```
For taew evaluation:
```
python evaluate.py --taew
```

## Further Usage Details for TAEW_NET
1. ```main.py```is the starting point of the code. It is runnable out-of-the-box once the datasets directory is downloaded and extracted. It also contains the full list of arguments for using the code.
2. ```utils/loader.py``` is used for loading the data and the labels. Labels are only available for the annotated part of the data.
3. ```utils/processor.py```contains the main training routine with forward and backward passes on the network, and parameter updates per iteration.
4. ```net/hapy.py``` contains the overall network and description of the forward pass.
