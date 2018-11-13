
from __future__ import print_function, division
import os
import argparse
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import datetime


train_csv = ""
test_csv = ""
outdir = ""
X_train = []
Y_train = np.array([])
X_test = []
Y_test = np.array([])       
max_workers=2 

def tempo():
    now = datetime.datetime.now()
    print ("Current date and time : ")
    print (now.strftime("%Y-%m-%d %H:%M:%S"))

def load_csvs():
    global train_csv
    global test_csv
    global outdir
    global X_train
    global Y_train
    global X_test
    global Y_test
    print("trainset loading ...")
    with open(train_csv, 'rt') as feature_file:
        for line in feature_file:
            # PATH - LABEL - N.tot.vis.feat - N.tot.txt.feat - features
            f_info = line.strip().split(",")[1:]

            y=int(f_info[0])

            f_info = [float(f) for f in  f_info[3:]]
            Y_train = np.append(Y_train, y)
            X_train += [f_info]
    print("testset loading ...")
    with open(test_csv, 'rt') as feature_file:
        for line in feature_file:
            f_info = line.strip().split(",")[1:]

            y=int(f_info[0])

            f_info = [float(f) for f in  f_info[3:]]
            Y_test = np.append(Y_test, y)
            X_test += [f_info]

def get_classifier_result(classifier_class, classifier_params):
    global train_csv
    global test_csv
    global outdir
    global X_train
    global Y_train
    global X_test
    global Y_test
    classifier = classifier_class(**classifier_params)
    classifier_name = classifier_class.__name__
    out_lines = []

    print('Trying', classifier_name, 'with', classifier_params)
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred, average='weighted')
    cm = confusion_matrix(Y_test, Y_pred)

    truths = Counter()
    for class_label in Y_test:
        truths[class_label] += 1

    true_positives = Counter()
    false_positives = Counter()
    for true_class, predictions in enumerate(cm):
        for predicted_class, count in enumerate(predictions):
            if predicted_class == true_class:
                true_positives[true_class] += count
            else:
                false_positives[true_class] += count

    precisions = {}
    recalls = {}
    for true_class, truth in truths.items():
        recalls[true_class] = true_positives[true_class] / truth
        precisions[true_class] = true_positives[true_class] / (true_positives[true_class] + false_positives[true_class])

    param_names = list(classifier_params.keys())
    param_names.sort()
    line = ','.join([classifier_name, 'F1-Score', 'Accuracy', 'Precision', 'Recall'] + param_names)
    out_lines.append(line)
    for class_label in truths:
        class_label = int(class_label)
        cols = [classifier_name, f1, accuracy]
        cols.append(str(np.mean(list(recalls.values()))))
        cols.append(str(np.mean(list(precisions.values()))))
        cols.extend([classifier_params[paramname] for paramname in param_names])
        cols.append(truths[class_label])
        cols.append(recalls[class_label])
        cols.append(precisions[class_label])
        cols.append('')
        cols.extend(list(cm[class_label]))
        cols = [str(col) for col in cols]

    cols = [classifier_name, f1, accuracy]
    cols.append(str(np.mean(list(recalls.values()))))
    cols.append(str(np.mean(list(precisions.values()))))
    cols.extend([classifier_params[paramname] for paramname in param_names])
    cols = [str(col) for col in cols]

    # now mean and F1 and Accuracy
    # cols = [classifier_name, 'MEAN()', '', '']
    # cols.extend(['' for paramname in param_names])
    # cols.append(str(np.mean(list(recalls.values()))))
    # cols.append(str(np.mean(list(precisions.values()))))
    out_lines.append(','.join(cols))
    out_lines.append('')
    return out_lines

def write_results(classifier, results):
    global train_csv
    global test_csv
    global outdir
    global X_train
    global Y_train
    global X_test
    global Y_test
    lines = []
    for result in results:
        lines.extend(result)
    with open(os.path.join(outdir, 'results_' + classifier.__name__ + '.csv'), 'wt') as f:
        for line in lines:
            f.write('{}\n'.format(line))


def write_results_temp(classifier, results,nome):
    global train_csv
    global test_csv
    global outdir
    global X_train
    global Y_train
    global X_test
    global Y_test
    lines = []
    for result in results:
        lines.extend(result)
    with open(os.path.join(outdir, 'results_' + nome+'.csv'), 'wt') as f:
        for line in lines:
            f.write('{}\n'.format(line))


def main(args):
    global train_csv
    global test_csv
    global outdir
    global X_train
    global Y_train
    global X_test
    global Y_test
    global max_workers
    train_csv = args.train
    test_csv = args.test
    outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    X_train = []
    Y_train = np.array([])
    X_test = []
    Y_test = np.array([])

    load_csvs()

    max_workers = int(args.n_threads)

    # Bayes
    if True:
        csv_result=outdir+'/results_' + GaussianNB.__name__ + '.csv'
        if os.path.exists(csv_result):
            print("{} esiste già!!".format(csv_result))
        else:
            tempo()
            result = get_classifier_result(classifier_class=GaussianNB, classifier_params={})
            write_results(GaussianNB, [result])

    

    # Decision Tree
    if True:
        csv_result=outdir+'/results_' + DecisionTreeClassifier.__name__ + '.csv'
        if os.path.exists(csv_result):
            print("{} already exists!!".format(csv_result))
        else:
            tempo()
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for criterion in ('gini', 'entropy'):
                    for splitter in ('best', 'random'):
                        future = executor.submit(get_classifier_result, DecisionTreeClassifier, {'criterion': criterion,
                                                                             'splitter': splitter,
                                                                             'max_depth': None,
                                                                             'min_samples_split': 2,
                                                                             'max_leaf_nodes': None})
                        futures.append(future)
                results = [future.result() for future in futures]
                write_results(DecisionTreeClassifier, results)
    
    # Random Forest
    if True:
        csv_result=outdir+'/results_' + RandomForestClassifier.__name__ + '.csv'
        if os.path.exists(csv_result):
            print("{} already exists!!".format(csv_result))
        else:
            tempo()
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for n_jobs in (1, 4):
                    future = executor.submit(get_classifier_result, RandomForestClassifier, {'n_jobs': n_jobs, 'n_estimators': 10})
                    futures.append(future)
                results = [future.result() for future in futures]
                write_results(RandomForestClassifier, results)


    # KNN
    if True:
        csv_result=outdir+'/results_' + KNeighborsClassifier.__name__ + '.csv'
        if os.path.exists(csv_result):
            print("{} already exists!!".format(csv_result))
        else:
            tempo()
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for n_neighbors in range(1, 26):
                    for metric in ('minkowski',):
                        for weight in ('uniform', 'distance'):
                            for algo in ('auto', 'ball_tree', 'kd_tree', 'brute'):
                                for p in (1, 2):
                                    future = executor.submit(get_classifier_result, KNeighborsClassifier, {'algorithm': algo,
                                                                                       'leaf_size': 30,
                                                                                       'metric': metric,
                                                                                       'metric_params': None,
                                                                                       'n_jobs': 1,
                                                                                       'n_neighbors': n_neighbors,
                                                                                       'p': p,
                                                                                       'weights': weight})  
                                    
                                    futures.append(future)

                            #Saving Partial Results
                            app=[future.result() for future in futures]
                            nome="KNN_{}_{}_{}".format(n_neighbors,metric,weight)
                            write_results_temp(svm.SVC, app,nome)

                results = [future.result() for future in futures]
                write_results(KNeighborsClassifier, results)

    
    # SVM
    if True:    
        csv_result=outdir+'/results_' + svm.SVC.__name__ + '.csv'
        if os.path.exists(csv_result):
            print("{} already exists!!".format(csv_result))
        else:
            gamma_C_pairs = []
            for C in (1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4):
                gamma_C_pairs.append(('auto', C))
            for C in (1e-3, 1e-2, 1e-1, 1, 1e1, 1e2):
                for gamma in (1e-3, 1e-2, 1e-1, 1, 1e1, 1e2):
                    gamma_C_pairs.append((gamma, C))

            tempo()
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for gamma, C in gamma_C_pairs:
                    for kernel in ('rbf', 'linear'):
                        for degree in (3,5):
                            future = executor.submit(get_classifier_result, svm.SVC, {'kernel': kernel,
                                                                  'C': C,
                                                                  'degree': degree,
                                                                  'gamma': gamma})
                            futures.append(future)
                    #Saving Partial Results
                    app=[future.result() for future in futures]
                    nome="SVM_{}_{}_{}_{}".format(kernel,gamma,C,degree)
                    write_results_temp(svm.SVC, app,nome)  
                results = [future.result() for future in futures]
                write_results(svm.SVC, results)  
    tempo()

if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description="")
    p.add_argument('-train', dest='train', action='store', default='', help='path train file *.csv')
    p.add_argument('-test', dest='test', action='store', default='', help='path test file *.csv')
    p.add_argument('-outdir', dest='outdir', action='store', default='.', help='path for output files')
    p.add_argument('-n_threads', dest='n_threads', action='store', default='.', help='number of threads')
    args = p.parse_args()
    main(args)
