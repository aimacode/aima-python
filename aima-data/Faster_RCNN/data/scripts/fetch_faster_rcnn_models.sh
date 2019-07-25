#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" && pwd )"
cd $DIR

NET=res101
FILE=voc_0712_80k-110k.tgz
# replace it with gs11655.sp.cs.cmu.edu if ladoga.graphics.cs.cmu.edu does not work
URL=http://ladoga.graphics.cs.cmu.edu/xinleic/tf-faster-rcnn/$NET/$FILE
CHECKSUM=cb32e9df553153d311cc5095b2f8c340

if [ -f $FILE ]; then
  echo "File already exists. Checking md5..."
  os=`uname -s`
  if [ "$os" = "Linux" ]; then
    checksum=`md5sum $FILE | awk '{ print $1 }'`
  elif [ "$os" = "Darwin" ]; then
    checksum=`cat $FILE | md5`
  fi
  if [ "$checksum" = "$CHECKSUM" ]; then
    echo "Checksum is correct. No need to download."
    exit 0
  else
    echo "Checksum is incorrect. Need to download again."
  fi
fi

echo "Downloading Resnet 101 Faster R-CNN models Pret-trained on VOC 07+12 (340M)..."

wget $URL -O $FILE

echo "Unzipping..."

tar zxvf $FILE

echo "Done. Please run this command again to verify that checksum = $CHECKSUM."
