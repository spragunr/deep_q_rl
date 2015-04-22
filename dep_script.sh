#!/bin/sh

echo "==>dependencies setup for deep_q_rl"

echo "==>updating current package..."
#sudo apt-get update

if [ ! -d "./cython" ]
then
echo "==>installing Cython ..."
git clone https://github.com/cython/cython
fi
cd ./cython
sudo python setup.py install
cd ..

echo "==>installing Theano ..."
# some dependencies ...
sudo apt-get install python-numpy python-scipy python-dev python-pip python-nose g++ libopenblas-dev git
sudo pip install Theano

if [ ! -d "./pylearn2" ]
then
echo "==>installing Pylearn2 ..."
git clone git://github.com/lisa-lab/pylearn2
fi
cd ./pylearn2
sudo python setup.py build
cd ..

if [ ! -d "./ALE" ]
then
echo "==>installing ALE ..."
git clone https://github.com/mgbellemare/Arcade-Learning-Environment
fi
mv Arcade-Learning-Environment ALE
cd ./ALE
#make USE_RLGLUE = 1 in makefile.unix
sed -i `s/USE_RLGLUE  := 0/USE_RLGLUE  := 1/g` makefile.unix
cp makefile.unix makefile
sudo make 
sudo cp ./ale /usr/bin
cd ..

if [ ! -d "./rlglue-py" ]
then
echo "==>installing rlglue-py ..."
git clone https://github.com/ctn-waterloo/rlglue-py
fi
cd ./rlglue-py
sudo python setup.py install
cd ..

echo "==>installing OpenCV..."
sudo apt-get install libgtk2.0-dev pkg-config
sudo apt-get install build-essential
sudo apt-get install cmake
sudo apt-get install wget
if [ ! -f "opencv-2.4.9.zip" ]
then
wget http://jaist.dl.sourceforge.net/project/opencvlibrary/opencv-unix/2.4.9/opencv-2.4.9.zip
unzip -q opencv-2.4.9.zip
fi

cd ./opencv-2.4.9
cmake .
sudo make
sudo make install


echo "==>All is done!"
