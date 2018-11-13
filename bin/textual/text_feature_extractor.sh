#!/usr/bin/env bash

CAFFE_PATH=../../txt/caffe-txt
RESULTS_PATH=../results/textual

for model in char2 dict2 ; do

    if [ ${model} = "char2" ]
    then
        iteration=171564
    else
        iteration=230046
    fi

    for layer in ip4 pool2 ; do
        for traintest in train test ; do
            DEPLOY=${CAFFE_PATH}/models/char-level_convnet/retail/deploy.prototxt
            SNAPSHOT=${RESULTS_PATH}/txt_features/snapshots/snapshot_${model}_iter_${iteration}.caffemodel
            DATASET=${RESULTS_PATH}/onehot/${model}_${traintest}.csv
            OUTPUT=${RESULTS_PATH}/txt_features/features_${model}_${traintest}_${layer}.csv

            ${CAFFE_PATH}/build/tools/featureextract ${DEPLOY}                \
                ${SNAPSHOT} \
                ${layer}                                                                        \
                ${DATASET}              \
                ${OUTPUT}
        done
    done
done

echo ""
echo "Generated:"
find ${RESULTS_PATH}/txt_features -iname 'features*.csv'