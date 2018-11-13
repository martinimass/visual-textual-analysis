#!/usr/bin/env bash

img_cnn=$1			#CNN for Visual Extractor
img_layer=$2		#CNN layer for Visual Extractor
txt_model=$3		#CNN Architecture for Textual Extractor [char2|dict2]
txt_layer=$4		#CNN layer for Textual Extractor
N_THREADS=$5		#How many parallel threads to run



function usage () {
    echo "Usage: $0 img_cnn img_layer txt_model txt_layer n_threads"
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
if [ -z "${N_THREADS}" ] ; then
    usage
fi

OUT_PATH=../results/final/${img_cnn}_${img_layer}_${txt_model}_${txt_layer}
TRAIN_PATH=../results/final/train_${img_cnn}_${img_layer}_${txt_model}_${txt_layer}_combined_dataset.csv
TEST_PATH=../results/final/test_${img_cnn}_${img_layer}_${txt_model}_${txt_layer}_combined_dataset.csv


mkdir -p ${OUT_PATH}

python final/final_classifier.py -train ${TRAIN_PATH} -test ${TEST_PATH} -outdir ${OUT_PATH} -n_threads ${N_THREADS} 2>&1 | tee ${OUT_PATH}/LOG.log