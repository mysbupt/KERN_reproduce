# This is the repo for paper: [Knowledge Enhanced Neural Fashion Trend Forecasting](https://arxiv.org/pdf/2005.03297.pdf)

## Requirements
1. OS: Ubuntu 16.04 or higher version
2. python3
3. Supported(tested) CUDA Versions: V7.5, V9.1, V10.2
4. python modules: refer to the modules in [requirements.txt](https://github.com/mysbupt/KERN_reproduce/blob/master/requirements.txt)

## Code Structure

1. The entry script is: train.py
2. The config file is: config.yaml
3. The script for data preprocess and dataloader: utility.py
4. The model folder: ./model/

## How to Run
1. Download the [dataset](https://drive.google.com/file/d/1E_gPmh6lIHyXEx3lRDWWKvm_8op5B82t/view?usp=sharing), decompress it and put it on the top directory: tar -zxvf dataset.tgz
Note that, the downloaded files include both datasets of GeoStyle and FIT.

2. Change the hyper-parameters in the configure file config.yaml. First, you should specify which dataset you want to use in the first line of config.yaml; second, you can use any pre-defined settings under each dataset (to make sure only keep one set of settings and comment all the others). Note that, all the settings correpsonding to the results in the companion paper's table 3/4 have been provided in the config.yaml, of which one is active and all the others are commented.

3. Run: train.py

4. Log/Runs: after running a eperiment under a certain setting, it will generate two log files: one is under ./log, which consists of the predictions, learned embeddings, screen outputs, and the learned model parameters; the other is ./runs, which is the log file that can be visualized by Tensorboard. You can go into the ./runs directory, execute "tensorboard --host="your host ip" --logdir=./", then you can see the training curves by a browser. Note that, you need to install the tensorboard to use this function.

5. Test using saved model: test.py.

- The best model will be saved under the folder ./log. You can run this script to load the saved model and produce the metrics score on the test set. Note that this script shares the same config file with train.py, i.e., config.yaml. In other words, you can change the config.yaml to decide which model to load and test. Specifically, we include in this repo the save models whose performance are reported in this companion paper. You can directly uncomment each corresponding part in the config.yaml and run the test.py to reproduce the results.

### Acknowledgement
This project is supported by the National Research Foundation, Prime Minister's Office, Singapore under its IRC@Singapore Funding Initiative.

<img src="https://github.com/mysbupt/KERN/blob/master/next.png" width = "297" height = "100" alt="next" align=center />
