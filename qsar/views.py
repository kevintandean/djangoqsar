import csv
import os
from django.http import HttpResponse
from django.shortcuts import render, render_to_response
import subprocess

from django import forms
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from joblib import Parallel, delayed
from sklearn import preprocessing, svm, metrics
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.externals import joblib as j
from sklearn.pipeline import make_pipeline
from qsar import learner
from qsar.learner import load_and_clean, save_model, split, pipeline_score
from qsarmodelling.settings import BASE_DIR
import chemspider
import numpy as np
import pandas as pd
import requests, xmltodict


class UploadFileForm(forms.Form):
    title = forms.CharField(max_length=50)
    file = forms.FileField()

class SearchForm(forms.Form):
    query = forms.CharField(max_length=100)

# Create your views here.
@csrf_exempt
def get_descriptor(request):
    form = SearchForm()
    score = ''
    if request.method == 'POST':
        form = SearchForm(request.POST)
        if form.is_valid():
                query = form.cleaned_data['query']
                # smiles = get_smiles(0,query)
                # if smiles == None:
                #     return HttpResponse('ERROR 500')

                # test_compound = pd.read_csv('test_compound.txt', sep='\t')
                name = 'test'
                # subprocess.call(['java','-jar','qsar/PaDEL-Descriptor/PaDEL-Descriptor.jar','-2d','-3d','-convert3d','-fingerprints','-dir','temp_smiles/'+name+'/','-file','temp_smiles/'+name+'/result.csv'])
                # test_compound = test_compound['col']
                # get_smiles_list(test_compound)
                # return 'sip'

                # data = Parallel(n_jobs=-1)(delayed(generate_smiles_descriptor)(query) for query in test_compound)
                score = generate_smiles_descriptor_score(query)
    return render_to_response('upload.html', {'form':form, 'score':score})

def run_fda(request):
    df = pd.read_csv('fda_list.txt', sep='\t')
    data = df['name']
    data_list = [item for item in data]
    result = {}
    count = 0
    for i in data_list:
        count += 1
        print count
        score = generate_smiles_descriptor_score(i)
        result[i]=score
        print score
    j.dump(result,'result.pkl')
    print result

    # print data_list
    # print'test'
    # result = Parallel(n_jobs=-1)(delayed(generate_smiles_descriptor_score(query) for query in data_list))
    # j.dump(result, 'result.pkl')
    # print result
    return HttpResponse('ok')

def get_smiles(request,name):
    smiles = chemspider.get_compound_smiles_id(name)
    if request == 0:
        print smiles
        return smiles
    else:
        return HttpResponse(smiles)

def get_smiles_list(array,name):
    table = {}
    path = 'temp_smiles/'+name+'/'+name+'.smi'
    filename = name+'.smi'
    dir_path = os.path.join('temp_smiles', name)
    try:
        os.makedirs(dir_path)
    except:
        pass
    f = open(os.path.join(dir_path, filename), 'wb')
    for item in array:
        smiles = get_smiles(0,item)
        if smiles == None:
            continue
        f.write(smiles+'    '+item+'\n')
        table[item] = smiles
    return 'temp_smiles/'+name+'/'

def get_descriptors_list(path, name):
    # path = 'temp_smiles/'+name+'/result.csv'
    subprocess.call(['java','-jar','qsar/PaDEL-Descriptor/PaDEL-Descriptor.jar','-2d','-3d','-convert3d','-retainorder','-fingerprints','-dir',path,'-file','temp_smiles/'+name+'/result.csv'])


    # generate_smiles_descriptor_score(query)


def generate_smiles_descriptor_score(query):
    smiles = chemspider.get_smiles(query)
    if smiles == '<h1>Page not found (404)</h1>\n':
        return '404'
    result = generate_descriptor(smiles,query,1000)
    return {'result':result,'name':query}

def generate_descriptor(smiles,name,n):
    filename = name+'.smi'
    dir_path = os.path.join('temp_smiles', name)
    # write_dir = os.path.join(dir_path,'descriptor')
    try:
        os.makedirs(dir_path)
    except:
        pass
    f = open(os.path.join(dir_path, filename), 'wb')
    # f = open(filename, 'w')
# w
#     def clean(x):
#         # print 'x',x
#         if np.isnan(float(x)):
#             # print 'nan',x
#             return 0
#         elif np.isfinite(float(x))==False:
#             # print 'infinity',x
#             return 0
#         # elif (x.dtype.char in np.typecodes['AllFloat']):
#             # print 'yes'
#             # return 0
#         elif float(x)>np.finfo(np.float64).max:
#             # print 'yes'
#             return 1
#         else:
#             return float(x)

    f.write(smiles)
    f.close()
    print os.getcwd()
    # subprocess.call(['java','-jar','qsar/PaDEL-Descriptor/PaDEL-Descriptor.jar','-2d','-3d','-fingerprints','-dir','temp_smiles/'+name+'/','-file','temp_smiles/'+name+'/result.csv'])
    # desc_file = open('temp_smiles/'+name+'/result.csv', 'r')
    # item1= desc_file.read()
    path = 'temp_smiles/'+name+'/result.csv'
    subprocess.call(['java','-jar','qsar/PaDEL-Descriptor/PaDEL-Descriptor.jar','-2d','-3d','-convert3d','-fingerprints','-dir','temp_smiles/'+name+'/','-file','temp_smiles/'+name+'/result.csv'])
    # desc_file = open(path, 'r')
    df = pd.read_csv(path, sep=',')
    x_sample = df.iloc[:, 1:n+1]
    x_sample = x_sample.apply(clean)
    # print x_sample
    x_sample.apply(clean)
    print 'applied again'
    clf = j.load('80d.pkl')

    y_pred = clf.predict(x_sample)
    # except:
    #     return 'fail'

    return y_pred[0]

    # print desc_file.read()

    # subprocess.call(['java','-jar','Users/kevintandean/Downloads/PaDEL-Descriptor/PaDEL-Descriptor.jar','-2d'])


        # file = request.FILES
        # print file
        # file = request.FILES["file"]
        # directory = os.path.join(BASE_DIR, 'temp_smiles_file')
        # fs = FileSystemStorage(location=directory)
        # print directory
        # fs.save('temp_file.smi', file)
#
#     subprocess.call(['java','-jar', '/Users/kevintandean/Downloads/PaDEL-Descriptor/PaDEL-Descriptor.jar', '-2d', '-dir', '/Users/kevintandean/djangoproject/qsarmodelling/temp_smiles_file/'
# , '-file', 'commandtestdjango5.csv'])




    # return render_to_response('upload.html', {'form': form})


def clean(x):
        # print 'x',x
        if np.isnan(float(x)):
            return 0
        elif np.isfinite(float(x))==False:
            # print 'infinity',x

            return 10000000000000000000000000000000000000000000000
        # elif (x.dtype.char in np.typecodes['AllFloat']):
            # print 'yes'
            # return 0
        elif float(x)>10000000000000000000000000000000000000:
            # print 'yes'
            return 1000000000000000000000000000000000000000
        else:
            return float(x)

def pipeline_score1(n, x_train,y_train,x_test,y_test,x_out):
    pipe = make_pipeline(preprocessing.StandardScaler(),ExtraTreesClassifier(), KernelPCA(n_components=n),svm.SVC())
    pipe.fit(x_train,y_train)
    predict = pipe.predict(x_test)
    score = metrics.accuracy_score(y_test, predict)
    predict_out = pipe.predict(x_out)
    return score, predict_out, pipe

def score_test(request):
    path = 'temp_smiles/test/result10.csv'
    df = pd.read_csv(path, sep=',')
    n = 1000
    df = df.iloc[:,1:n+1]
    df.fillna(0)
    df = df.applymap(clean)
    rows_with_error = df.apply(
           lambda row : any([ e == '#NAME?' or np.isfinite(float(e))==False for e in row ]), axis=1)

    df = df[~rows_with_error]
    # df = df.applymap(clean)
    descriptors = load_and_clean('qsar/all data full descriptors.txt',15,n+15)
    x_train, x_test, y_train, y_test = split(descriptors, 0.3)
    score, pred, pipe = pipeline_score1(110, x_train,y_train,x_test,y_test,df)
    # j.dump(pipe, '80d.pkl')
    count = 0
    for item in pred:
        if int(item) == 1:
            count +=1
    print score
    print count
    print 'total',len(pred)
    print pred
    print 'dumped'



    # clf = joblib.load('pipe.pkl')
    # y_pred = clf.predict(df)
    # print y_pred


            #
        # if form.is_valid():
        #     handle_uploaded_file(request.FILES['file'])
        #     return HttpResponseRedirect('/success/url/')
    # return render_to_response('upload.html', {'form': form})
#
# subprocess.call(['java','-jar', '/Users/kevintandean/Downloads/PaDEL-Descriptor/PaDEL-Descriptor.jar', '-2d', '-dir', '/Users/kevintandean/Downloads/'
# , '-file', 'commandtestdjango2.csv'])
    #

    # return 'ok'

# subprocess.call(['java','-jar', '/padelzip/PaDEL-Descriptor/PaDEL-Descriptor.jar', '-2d', '-dir', '/home/ec2-user/'
# , '-file', 'commandtestdjango2.csv'])
# subprocess.call(['java','-jar', '/padelzip/PaDEL-Descriptor/PaDEL-Descriptor.jar', '-2d', '-dir', '/home/ec2-user/'
# , '-file', 'commandtestdjango2.csv'])
#
# java -jar /padelzip/PaDEL-Descriptor/PaDEL-Descriptor.jar