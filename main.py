#coding: utf-8

from core import Core
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import caching

t = Core.getTwitter()
tuning = True

if __name__ == '__main__':
    print("Start Learning")

    train_df = pd.read_pickle("soma_goods_train.df")

    vectorizer = CountVectorizer(dtype='bool')
    #vectorizer = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'[\S]{1,}', ngram_range=(1, 2), dtype='bool', use_idf=1, smooth_idf=1, sublinear_tf=1)

    d_list = []
    cate_list = []
    for each in train_df.iterrows():
        cate = ";".join([each[1]['cate1'], each[1]['cate2'], each[1]['cate3']])
        d_list.append(each[1]['name'])
        cate_list.append(cate)

    cate_dict = dict(zip(list(set(cate_list)), range(len(set(cate_list)))))

    print("Load Features..")

    y_list = []
    for each in train_df.iterrows():
        cate = ";".join([each[1]['cate1'], each[1]['cate2'], each[1]['cate3']])
        y_list.append(cate_dict[cate])

    n_list = []
    for d in d_list:
        n_list.append(Core.proc(d))

    print("Vectorizing..")

    x_list = vectorizer.fit_transform(n_list)

    svc_param = {'C': np.logspace(-2, 0, 20)}

    if not tuning:
        print("Grid Searching..")

        gs_svc = GridSearchCV(LinearSVC(loss='squared_hinge'), svc_param, cv=10, n_jobs=5, verbose=10)
        gs_svc.fit(x_list, y_list)

        print(gs_svc.best_params_, gs_svc.best_score_)

        clf = LinearSVC(C=gs_svc.best_params_['C'])  # // C= 랜덤으로 돌림 -3 ~ 1
        clf.fit(x_list, y_list)
    else:
        clf = LinearSVC(C=float("0.059519279617756058"))
        clf.fit(x_list, y_list)

    #0.029519279617756058 -> 72.4898
    #0.059519279617756058 -> 72.5714
    #0.030249961544146747

    #joblib.dump(clf, 'dumps\\classify.model', compress=3)
    #joblib.dump(cate_dict, 'dumps\\cate_dict.dat', compress=3)
    #joblib.dump(vectorizer, 'dumps\\vectorizer.dat', compress=3)

    caching.run(clf, cate_dict, vectorizer, d_list, cate_list)

    print("Done")
