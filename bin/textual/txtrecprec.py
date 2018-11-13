#!/usr/bin/env python
from __future__ import print_function, division
from collections import defaultdict, Counter
import sys
import numpy as np


#cid2cn = {0: 'positive', 1: 'neutral', 2: 'negative'}
cid2cn = {0: 'SOOS', 1: 'Normal', 2: 'Promotion'}


def recall(stats, category):
    tp = stats[category]['tp']
    fn = stats[category]['fn']
    if tp == 0:
        return 0
    return tp / (tp + fn)


def precision(stats, category):
    tp = stats[category]['tp']
    fp = stats[category]['fp']
    if tp == 0:
        return 0
    return tp / (tp + fp)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: {} classification.csv output.csv'.format(sys.argv[0]))
        sys.exit()

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]

    stats = defaultdict(Counter)
    confusion = defaultdict(Counter)

    count = 0
    with open(input_csv, 'rt') as f:
        for line in [line.strip() for line in f.readlines()[1:]]:
            record, label, prediction = [int(col) for col in line.split(',')]
            label = cid2cn[label]
            prediction = cid2cn[prediction]
            count += 1
            stats[label]['truth'] += 1
            if label == prediction:
                stats[label]['tp'] += 1
            else:
                stats[label]['fn'] += 1  # false negative for the label class: we missed this one
                stats[prediction]['fp'] += 1  # false positive for the predicted class
            confusion[label][prediction] += 1

    # P = tp / (tp + fp)
    # R = tp / (tp + fn)
    with open(output_csv, 'wt') as f:
        f.write('{:10},{:10},{:10},{:10},,{:10},{:10},{:10}\n'.format('Category', 'Truth', 'Recall', 'Precision',
                                                                      'SOOS', 'Normal', 'Promotion'))
        recalls = []
        precisions = []
        for category in ['SOOS', 'Normal', 'Promotion']:
            truth = stats[category]['truth']
            r = recall(stats, category)
            p = precision(stats, category)
            recalls.append(r)
            precisions.append(p)
            f.write('{:10},{:10},{:10.6f},{:10.6f},,{:10},{:10},{:10}\n'.format(category, truth, r, p,
                                                                          confusion[category]['SOOS'],
                                                                          confusion[category]['Normal'],
                                                                          confusion[category]['Promotion']
                                                                          ))
        f.write('{:10},{:10},{:10},{:10}\n'.format('MEAN()', '', np.mean(recalls), np.mean(precisions)))
