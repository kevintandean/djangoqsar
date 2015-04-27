import time

__author__ = 'kevintandean'
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import svm, metrics, preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import chi2, f_classif
from joblib import Parallel, delayed
from sklearn.decomposition import KernelPCA
from sklearn.pipeline import make_pipeline


def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print '%r %2.2f sec' % \
              (method.__name__, te-ts)
        return result

    return timed

def load_and_clean(path, start, descriptors_n):
    df = pd.read_csv(path, sep='\t')
    descriptors = df.iloc[:, start:descriptors_n]
    # descriptors = descriptors.loc[df['LOG BB'] == 'BBB+'] + descriptors.loc[df['LOG BB'] == 'BBB-']
    # mask = df['LOG BB'] == 'BBB+'
    # mask2 = df['LOG BB'] == 'BBB-'
    # mask = mask + mask2
    # descriptors = descriptors[mask]
    # print descriptors.shape

    logbb = df['LOG BB']

    def binarize(x):
        if x == 'BBB+':
            return 1
        elif x == 'BBB-':
            return 0
        elif x > 0:
            return 1
        elif x <= 0:
            return 0


    logbb = logbb.apply(binarize)
    positive=0
    negative=0
    for i in logbb:
        if i == 1:
            positive +=1
        elif i == 0:
            negative +=1

    descriptors = descriptors.join(logbb)
    descriptors.fillna(0)

    rows_with_error = descriptors.apply(
           lambda row : any([ e == '#NAME?' or np.isfinite(float(e))==False for e in row ]), axis=1)

    # print rows_with_error
    descriptors = descriptors[~rows_with_error]



    descriptors = descriptors.applymap(lambda x: float(x))

    return descriptors

# descriptors = load_and_clean('discretedata.txt',2,2000)



def split(data, size):
    grouped = data.groupby('LOG BB')
    bbb_neg = grouped.get_group(0.0)
    bbb_pos = grouped.get_group(1.0)
    descriptor_n = bbb_neg.shape[1]

    # descriptor_n = 2756
    # descriptor_n = 30
    n = bbb_neg.shape[0]
    # n = 850
    # n = 0


    x_pos = bbb_pos.iloc[:n,0:descriptor_n-1].values
    y_pos = bbb_pos.iloc[:n,descriptor_n-1:descriptor_n].values
    x_pos_train, x_pos_test, y_pos_train, y_pos_test = train_test_split(x_pos, y_pos, test_size=size, random_state=100)

    x_neg = bbb_neg.iloc[:,0:descriptor_n-1].values
    y_neg = bbb_neg.iloc[:,descriptor_n-1:descriptor_n].values
    x_neg_train, x_neg_test, y_neg_train, y_neg_test = train_test_split(x_neg, y_neg, test_size=size, random_state=100)

    x_train = np.append(x_pos_train, x_neg_train, axis = 0)
    y_train = np.append(y_pos_train, y_neg_train, axis = 0)
    x_test = np.append(x_pos_test, x_neg_test, axis = 0)
    y_test = np.append(y_pos_test, y_neg_test, axis = 0)


    return x_train, x_test, y_train, y_test

def scorer(y_test,y_pred):
    true_negative = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for i in range(0,len(y_test)):
        # print y_pred[i], y_test[i]
        if int(y_pred[i]) == int(y_test[i]):
            if y_pred[i] == 0:
                true_negative += 1
            elif y_pred[i] == 1:
                true_positive += 1
        elif int(y_pred[i]) != int(y_test[i]):
            if y_pred[i] == 0:
                false_negative += 1
            elif y_pred[i] == 1:
                false_positive += 1
    # print true_positive, true_negative, false_positive, false_negative
    sensitivity = float(true_positive)/(true_positive+false_negative)
    specificity = float(true_negative)/(true_negative+false_positive)

    return sensitivity, specificity

from sklearn.decomposition import PCA
# pca = PCA(copy = True, n_components=10)
# pca.fit(x_train)
# x_train_pca = pca.transform(x_train)
# x_train_pca.shape
# x_test_pca = pca.transform(x_test)
# x_test_pca.shape

def reduce_to(x_train,x_test,n):
    pca = PCA(copy=True, n_components = n)
    pca.fit(x_train)
    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)
    return x_train_pca, x_test_pca, n

# n, x_train_pca, x_test_pca = reduce_to(x_train,x_test,15)
# print x_train_pca.shape, n

def optimize_pca(x_train,x_test,y_train,y_test,n_test):
    d = {}
    for i in range(1,n_test):
        print i
        x_train_pca, x_test_pca, n= reduce_to(x_train,x_test,i)
        clf3 = svm.SVC()
        clf3.fit(x_train_pca, y_train)
        score = clf3.score(x_test_pca,y_test)
        d[n] = score

    return d



def find_max(dict,item):
    key = 0
    max_value = {item:0}
    for k,v in dict.iteritems():
#         print 'item',v[item]
#         print 'max',max_value
        if v[item]>max_value[item]:
            max_value=v
            key = k
    return {'n':key, 'score':max_value}


def optimize(n_start, n_end):
    data = {}
    for i in range(n_start,n_end):
#         print i
        pipe = make_pipeline(preprocessing.StandardScaler(),ExtraTreesClassifier(), KernelPCA(n_components=i),svm.SVC())
        pipe.fit(x_train,y_train)
        predict = pipe.predict(x_test)
        score = metrics.accuracy_score(y_test, predict)
        sensitivity, specificity = scorer(predict,y_test)
        data[i] = {'tn':specificity, 'tp':sensitivity,'acc':score}
    max_tn = find_max(data,'tn')
    max_tp = find_max(data,'tp')
    max_acc = find_max(data,'acc')
    return {'max_acc':max_acc}

def pipeline_score(n, x_train,y_train,x_test,y_test):
    pipe = make_pipeline(preprocessing.StandardScaler(),ExtraTreesClassifier(), KernelPCA(n_components=n),svm.SVC())
    pipe.fit(x_train,y_train)
    predict = pipe.predict(x_test)
    score = metrics.accuracy_score(y_test, predict)
    return {'n':n,'score':score}

def optimize_parallel(n_start, n_end, x_train,y_train,x_test,y_test):
    data = Parallel(n_jobs=-1)(delayed(pipeline_score)(n,x_train,y_train,x_test,y_test) for n in range(n_start,n_end))
    max = 0
    max_key = 0
    for i in data:
        if i['score']>max:
            max=i['score']
            max_key = i['n']
    return {'n':max_key, 'score':max}


@timeit
def common(n,x_train,y_train,x_test,y_test):
    result = Parallel(n_jobs=-1)(delayed(optimize_parallel)(70,130,x_train,y_train,x_test,y_test) for i in range(n))
    # for i in range(0,n):
    #     result[i] = optimize(70,120)
    return result


def find_common(data):
    result = {}
    max_key = 0
    max_value = 0
    for k,v in data.iteritems():
        n = v['max_acc']['n']
        score = v['max_acc']['score']
        if n in data:
            result[n]+=1
            if result[n]>max_value:
                max_key = n
                max_value = score
        else:
            result[n] = 1
    return {'n':max_key, 'score':max_value}


def split_train(data, n):
    x_train, x_test, y_train, y_test = split(data, n)
    result = common(1,x_train,y_train,x_test,y_test)
    return {'valid_fraction':n, 'score':result}

def optimize_train(data, n_list):
    result = Parallel(n_jobs=-1)(delayed(split_train)(data, n) for n in n_list)
    return result

from sklearn.externals import joblib

def save_model(x_train,y_train, n):
    pipe = make_pipeline(preprocessing.StandardScaler(),ExtraTreesClassifier(), KernelPCA(n_components=n),svm.SVC())
    pipe.fit(x_train,y_train)
    joblib.dump(pipe, 'pipe.pkl')


if __name__ == '__main__':
    descriptors = load_and_clean('all data full descriptors.txt',15,2770)
    x_train, x_test, y_train, y_test = split(descriptors, 0.3)
    clf = joblib.load('pipe.pkl')
    score = clf.score(x_test,y_test)
    print score
    # save_model(x_train,y_train,73)
    # result = common(100,x_train,y_train,x_test,y_test)
    # result = optimize_train(descriptors,[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
    # print result

    # data = common(100,x_train,y_train,x_test,y_test)
    # # result = find_common(data)
    # print data
# x_train_pca, x_test_pca, n= reduce_to(x_train,x_test,110)

from sklearn.lda import LDA


# scaler = preprocessing.StandardScaler().fit(x_train)
# x_train_scale = scaler.transform(x_train)
# x_test_scale = scaler.transform(x_test)
#
# tree = ExtraTreesClassifier()
# tree_estimator = tree.fit(x_train_scale, y_train)
# x_train_scale = tree_estimator.transform(x_train_scale)
# x_test_scale = tree_estimator.transform(x_test_scale)
#
# print 'shape after tree', x_train_scale.shape
# x_train_pca, x_test_pca, n= reduce_to(x_train_scale,x_test_scale,110)
#
# reduce_lda = LDA(n_components=50)
# reduce_lda.fit(x_train_pca, y_train)
# x_train_lda = reduce_lda.transform(x_train_pca)
# x_test_lda = reduce_lda.transform(x_test_pca)


def optimize_svm(xtrain, ytrain, xtest, ytest, c_range):
    data = {}
    for i in c_range:
        clf = svm.SVC(C=i)
        clf.fit(xtrain,ytrain)
        score = clf.score(xtest, ytest)
        print i, score
        y_pred = clf.predict(xtest)
        sensitivity, specificity = scorer(ytest,y_pred)
        print 'sensitivity:', sensitivity
        print 'specificity:', specificity
        data[i] = score
    return data


#
# def split_train(n):
#     x_train, x_test, y_train, y_test = split(descriptors, 0.3)



# result = optimize_svm(x_train_lda, y_train, x_test_lda, y_test, [1000000])
# print result
# print x_train
# print x_test.shape, y_train.shape



# estimator = svm.SVC(kernel='linear')
# selector = RFE(estimator,7)
# selection = selector.fit(x_train,y_train)
# x_train_sel = selection.transform(x_train)
# x_test_sel = selection.transform(x_test)
# print x_train_sel.shape
#
# # import pickle
# # s = pickle.dumps(selector)
#
# from sklearn.externals import joblib
# joblib.dump(selector, 'filename.pkl')



# forest = RandomForestClassifier()
# forest.fit(x_train,y_train)
# print forest.score(x_test,y_test)
#
#





#
# for i in [0.1,0.2,0.3,0.4]:
#     x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=i, random_state=42)
#     # print x_train
#     # print y_train
#     # print x_test
#     # print y_test
#
#
#     clf.fit(x_train,y_train)
#     forest.fit(x_train,y_train)
#     print i
#     print clf.score(x_test,y_test)
#     print forest.score(x_test,y_test)