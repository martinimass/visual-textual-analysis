#!/usr/bin/env bash


model=$1
flagC=0
flagD=0

if [ ${model} = "char2" ]
then
	flagC=1
else
	if [ ${model} = "dict2" ]
	then
		flagD=1
	else
		if [ -z "$model" ] ; then
			flagC=1
			flagD=1
		fi
	fi
fi

CAFFE_PATH=../../txt/caffe-txt
RESULTS_PATH=../results/textual
LOG_PATH=${RESULTS_PATH}/txt_features/snapshots

mkdir -p ${LOG_PATH}


#CHAR2
if [ ${flagC} = 1 ] ; then

	${CAFFE_PATH}/build/tools/trainer ${RESULTS_PATH}/onehot/char2_train.csv \
	                ${RESULTS_PATH}/onehot/char2_test.csv  \
	                ${CAFFE_PATH}/models/char-level_convnet/retail/solver_char2.prototxt                                 \
	                ${CAFFE_PATH}/models/char-level_convnet/model_iter_300000.caffemodel 2>&1 | tee ${LOG_PATH}/train_char2.log
fi

#DICT2
if [ ${flagD} = 1 ] ; then

	${CAFFE_PATH}/build/tools/trainer ${RESULTS_PATH}/onehot/dict2_train.csv \
	                ${RESULTS_PATH}/onehot/dict2_test.csv  \
	                ${CAFFE_PATH}/models/char-level_convnet/retail/solver_dict2.prototxt                                 \
	                ${CAFFE_PATH}/models/char-level_convnet/model_iter_300000.caffemodel 2>&1 | tee ${LOG_PATH}/train_dict2.log
fi
