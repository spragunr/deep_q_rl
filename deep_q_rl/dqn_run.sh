#!/bin/sh

# you can set USE_GPU=1 to use gpu to train ,otherwise only cpu 
export USE_GPU=1
#export USE_GPU=1
python ale_run.py --exp_pref data
unset USE_GPU
