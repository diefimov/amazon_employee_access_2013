Out solution for Amazon.com - Employee Access Challenge 
https://www.kaggle.com/c/amazon-employee-access-challenge
(3rd place, team Dmitry&Leustagos, members: Lucas Silva, Dmitry Efimov)
===========================

# License
Copyright [2013] [Dmitry Efimov, Lucas Silva]
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

# How to use it
1. The hardware / OS platform you used
Windows 7 Professional x64 or Ubuntu (tested on 12.04)

2. Any necessary 3rd-party software (+ installation steps)

R 2.15.3 (http://www.r-project.org/)
Packages: data.table, gbm, randomForest, glmnet, parallel, doSNOW, foreach, Metrics, cvTools, lme4, rlecuyer

Python 2.6 (http://www.python.org/download/releases/2.6/) 
Packages: numpy, sklearn, scipy, itertools, pandas, sys, random, time

libFM (http://www.libfm.org/)

# How to train models and make predictions on a new test set

The file list with description

1) Files for models training:
[01]train.gbm.freq.hom2.R
[01]train.gbm.freq.hom3.R
[01]train.gbm.lme.R
[01]train.gbm.occurs.R
[01]train.glmnet.R
[01]train.glmnet2.R
[01]train.libfm.R
[01]train.lr.R
[01]train.lr2.R
[01]train.rf.freq.R
[02]train.gbm.freq.hom5.R
[02]train.gbm.occurs.xor.libfm.R

2) File for the final ensembling
[03]ensembling.R

3) Python code for logistic regression
logistic_regression.py

4) File with helper functions
fn.base.R

# To calculate model

1) Copy initial train.csv and test.csv to the folder data.
2) Copy libFM binary file (libfm.exe) to the folder libfm.
3) Run all R files with prefix [01] (they can be run in parallel).
To run any R file, open it in R GUI, set directory to the file location and run the code.
4) Run all R files with prefix [02]
5) Run [03]ensembling.R
6) The file prediction.csv contains final prediction for the test set.
