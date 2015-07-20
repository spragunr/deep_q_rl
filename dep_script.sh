#!/bin/bash

echo "==>dependencies setup for deep_q_rl"

echo "==>updating current package..."
#sudo apt-get update

echo "==>installing Cython..."
sudo apt-get install cython

echo "==>installing OpenCV..."
sudo apt-get install python-opencv

echo "==>installing Matplotlib..."
sudo apt-get install python-matplotlib python-tk

echo "==>installing Theano ..."
# some dependencies ...
sudo apt-get install python-numpy python-scipy python-dev python-pip python-nose g++ libopenblas-dev git
sudo pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git


# Packages below this point require downloads. 
mkdir build
cd build

if [ ! -d "./pylearn2" ]
then
echo "==>installing Pylearn2 ..."
# dependencies...
sudo apt-get install libyaml-0-2 python-six
git clone git://github.com/lisa-lab/pylearn2
fi
cd ./pylearn2
sudo python setup.py develop
sudo rm -r ~/.theano
cd ..

if [ ! -d "./ALE" ]
then
echo "==>installing ALE ..."

# dependencies ...
sudo apt-get install libsdl1.2-dev libsdl-gfx1.2-dev libsdl-image1.2-dev cmake

git clone https://github.com/mgbellemare/Arcade-Learning-Environment ALE
cd ./ALE
cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF .
make
sudo pip install .
cd ..
fi

if [ ! -d "./Lasagne" ]
then
echo "==>installing Lasagne ..."

git clone https://github.com/Lasagne/Lasagne.git
cd ./Lasagne
sudo python setup.py install
fi

echo "==>All done!"
