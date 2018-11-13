#!/usr/bin/env bash

img_cnn=$1			#CNN for Visual Extractor
img_layer=$2		#CNN layer for Visual Extractor
txt_model=$3		#CNN Architecture for Textual Extractor [char2|dict2]
txt_layer=$4		#CNN layer for Textual Extractor

OUT_PATH=../results/final

function usage () {
    echo "Usage: $0 img_cnn img_layer txt_model txt_layer"
    exit
}

if [ -z "${img_cnn}" ] ; then
    usage
fi
if [ -z "${img_layer}" ] ; then
    usage
fi
if [ -z "${txt_model}" ] ; then
    usage
fi
if [ -z "${txt_layer}" ] ; then
    usage
fi

python final/feature_combiner.py ${img_cnn} ${img_layer} ${txt_model} ${txt_layer} ${OUT_PATH}
