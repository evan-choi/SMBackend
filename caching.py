from core import Core
from sklearn.externals import joblib


def run(clf, cate_dict, vectorizer, train_datas, train_cates):
    cate_id_name_dict = {}

    for item in cate_dict.items():
        cate_id_name_dict[item[1]] = item[0]

    print("Start Caching..")

    test_preset = {}
    with open('data_a.txt', 'r') as f:
        for l in f.readlines():
            cl = l.strip('\n').replace('�', '')

            if cl in train_datas:
                test_preset[cl] = train_cates[train_datas.index(cl)]
                continue

            if (cl in test_preset):
                continue

            pred = clf.predict(vectorizer.transform([Core.proc(cl, 4)]))[0]
            test_preset[cl] = cate_id_name_dict[pred]

    joblib.dump(test_preset, 'dumps\\test_preset.dat', compress=3)
