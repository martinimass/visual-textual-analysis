#!/usr/bin/env bash
folder="../results/textual"

# char2 net
for d in train test ; do
    sudo THEANO_FLAGS='device=cpu' /usr/bin/python2.7 textual/batch_charnet.py $folder/$d
done

# dict2 net
for d in train test ; do
    sudo THEANO_FLAGS='device=cpu' /usr/bin/python2.7 textual/batch_dictnet.py $folder/$d
done
