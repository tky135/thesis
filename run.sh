#!/bin/sh
eval "$(~/miniconda3/bin/conda shell.bash hook)" 
conda activate plaid
mkdir $1
cd $1
cp ../train.py .
cp -r ../lib .
cp -r ../misc .
ln -s ../iREPA
export PYTHONPATH=.
python train.py
