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
sudo pip install Theano


# Packages below this point require downloads. 
mkdir build
cd build

if [ ! -d "./pylearn2" ]
then
echo "==>installing Pylearn2 ..."
# dependencies...
sudo apt-get install libyaml-0-2
git clone git://github.com/lisa-lab/pylearn2
fi
cd ./pylearn2
sudo python setup.py develop
sudo rm -r ~/.theano
cd ..


echo "==>installing rlglue3.04..."
sudo apt-get install wget
if [ ! -d "./rlglue-3.04" ]
then
wget http://rl-glue-ext.googlecode.com/files/rlglue-3.04.tar.gz
tar -xvf rlglue-3.04.tar.gz
fi
cd ./rlglue-3.04
./configure
make
sudo make install
cd ..


if [ ! -d "./rlglue-py" ]
then
echo "==>installing rlglue-python-codec ..."
git clone https://github.com/ctn-waterloo/rlglue-py
fi
cd ./rlglue-py
sudo python setup.py install
cd ..


if [ ! -d "./ALE" ]
then
echo "==>installing ALE ..."

# dependencies ...
sudo apt-get install  libsdl1.2-dev libsdl-gfx1.2-dev libsdl-image1.2-dev

git clone https://github.com/mgbellemare/Arcade-Learning-Environment
mv Arcade-Learning-Environment ALE
cd ./ALE
#make USE_RLGLUE = 1 and USE_SDL = 1 in makefile.unix
sed -i 's/USE_RLGLUE  := 0/USE_RLGLUE  := 1/g' makefile.unix
sed -i 's/USE_SDL     := 0/USE_SDL     := 1/g' makefile.unix
cp makefile.unix makefile
sudo make 
sudo cp ./ale /usr/bin
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

