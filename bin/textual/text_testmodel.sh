#!/usr/bin/env bash

CAFFE_PATH=../../txt/caffe-txt
RESULTS_PATH=../results/textual

traintest=test
for model in char2 dict2 ; do
	if [ ${model} = "char2" ]
	then
		iteration=171564
	else
		iteration=230046
	fi
	echo "EVALUATE ${model} model - iteration=${iteration}"
	
	DEPLOY=${CAFFE_PATH}/models/char-level_convnet/retail/deploy.prototxt
	SNAPSHOT=${RESULTS_PATH}/txt_features/snapshots/snapshot_${model}_iter_${iteration}.caffemodel
	DATASET=${RESULTS_PATH}/onehot/${model}_${traintest}.csv
	CSV_RESULTS=${RESULTS_PATH}/txt_features/results_${traintest}_${model}_${iteration}.csv

	${CAFFE_PATH}/build/tools/txtclassify ${DEPLOY}    \
	             ${SNAPSHOT}    \
	             ${DATASET} 	\
	             ${CSV_RESULTS}

	CSV_PRECISION=${RESULTS_PATH}/txt_features/precision_recall_${traintest}_${model}_${iteration}.csv

	python textual/txtrecprec.py ${CSV_RESULTS} ${CSV_PRECISION}

	#serve solo per l'output finale
	if [ ${model} = "char2" ]
	then
		save1=${CSV_PRECISION}
	else
		save2=${CSV_PRECISION}
	fi

done
echo "Generated: ${save1}"
cat ${save1}
echo "Generated: ${save2}"
cat ${save2}