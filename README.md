# RST Discourse Parsing with Coreference Information
This repository contains the code for our experiments on improving RST discourse parsing with various approaches for integrating information from a coreference resolver.

## Running experiments
### Coreference resolver
First, set up the coreference resolver as described [here](https://github.com/grig-guz/ubc-coref). In its main folder, run 
 ```bash
 python -m pip install -e .
```
to install it as library.

### Preparing data
Place the contents of training and testing portions of RST-DT inside the data/data_dir folder. It should look like this:
 ```bash
data/data_dir/train_dir/*
data/data_dir/test_dir/*
src/
```

### Preprocessing
  This project relies on Stanford CoreNLP toolkit to preprocess the data. You can download from [here](http://stanfordnlp.github.io/CoreNLP/index.html) and put the file [run_corenlp.sh](./run_corenlp.sh) into the CoreNLP folder. Then use the following command to preprocess both the data in train_dir and in test_dir:
    
  ```
  python preprocess.py --data_dir DATA_DIR --corenlp_dir CORENLP_DIR
  ```
Next, run the following to generate the action/relation maps, coreference clusters and train/dev split:
  ```
  python main.py --prepare --train_dir TRAIN_DIR --pretrained_coref_path PATH
  ```
where ```--pretrained_coref_path``` specifies the path to pretrained coreference model, which can be downloaded from [here](TODO)
  ### Training
You need to specify model type:
* 0 for the baseline model (no coreference)
* 1 for the model utilizing coreference features
* 2 for multitask model with coreference features
* 3 for multitask model without coreference features
 ```
python main.py --train --model_name YOUR_MODEL_NAME --model_type NUM --pretrained_coref_path PATH
```
The models are saved at ```data/model``` directory, and the training will be resumed from the last epoch if it was interrupted.

### Testing
Similar to above:
 ```
python main.py --eval --eval_dir ../data/data_dir/test_dir/ --model_name YOUR_MODEL_NAME --model_type NUM --pretrained_coref_path PATH
 ```


