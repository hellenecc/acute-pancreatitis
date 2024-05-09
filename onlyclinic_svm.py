import torch as t
from torch.utils import data
import os
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from utilities import *
from PIL import Image
import numpy as np
import torchvision
import torchvision.utils as utils
from torch.utils.data import random_split
import random
import shutil
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score, roc_auc_score
import pandas as pd
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from scipy.stats import norm
import numpy as np
from sklearn import preprocessing


from random import randint
from sklearn import svm

import os
gpuid = '0,1,2,3,4,5,6,7,8,9'
os.environ['CUDA_VISIBLE_DEVICES'] = gpuid

zscore = preprocessing.StandardScaler()
def softmax(x, axis=1):
	row_max = x.max(axis=axis)
	row_max = row_max.reshape(-1, 1)
	hatx = x - row_max
	hatx_exp = np.exp(hatx)
	hatx_sum = np.sum(hatx_exp, axis=axis, keepdims=True)
	s = hatx_exp / hatx_sum
	return s


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        pass
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    ax.set_ylim(len(classes)-0.5, -0.5)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def datasetcv_train(trainx, trainy, clinicdata_train=[], fold=None):

    xlsx_feature = clinicdata_train[fold]
    train_path = trainx
    label = trainy
    feature = xlsx_feature[xlsx_feature['UID'] == train_path]

    clinicfeature = feature.iloc[:, 1:-1]

    clinicfeature = np.float32(clinicfeature)
    return clinicfeature, label


def datasetcv_val(valx, valy, clinicdata_val=[], fold=None):
    xlsx_feature = clinicdata_val[fold]

    val_path = valx
    label = valy
    feature = xlsx_feature[xlsx_feature['UID'] == val_path]
    clinicfeature = feature.iloc[:, 1:-1]

    clinicfeature = np.float32(clinicfeature)
    return clinicfeature, label





def cross_val(k):

    clinicdata_train = []
    clinicdata_val = []
    valpath = []
    trainpath = []
    tmp = pd.DataFrame()
    clilevel1 = []
    clilevel2 = []
    clilevel3 = []

    clinicfile = '..'  # ID features target
    clinicdata = pd.read_excel(clinicfile)
    ids = clinicdata['UID'].tolist()
    random.shuffle(ids)
    c_name = clinicdata.columns
    c_namedata = c_name[1:-1]
    fortrain = pd.DataFrame()
    fortrainid = pd.DataFrame()

    for id in ids:
        tmp = clinicdata[clinicdata['UID'] == id]
        if tmp['sure'].values == 0:
            clilevel1.append(id)

        elif tmp['sure'].values == 1:
            clilevel2.append(id)
        else:
            clilevel3.append(id)
    c_name = clinicdata.columns
    c_namedata = c_name[1:-1]

    for i in range(k):
        val = []
        train = []
        ZS = preprocessing.StandardScaler()

        nval1 = len(clilevel1) // k
        val.extend(clilevel1[nval1 * i:nval1 * (i + 1)])
        train.extend(clilevel1[:nval1 * i] + clilevel1[(i + 1) * nval1:])
        nval2 = len(clilevel2) // k
        val.extend(clilevel2[nval2 * i:nval2 * (i + 1)])
        train.extend(clilevel2[:nval2 * i] + clilevel2[(i + 1) * nval2:])
        nval3 = len(clilevel3) // k
        val.extend(clilevel3[nval3 * i:nval3 * (i + 1)])
        train.extend(clilevel3[:nval3 * i] + clilevel3[(i + 1) * nval3:])
        valpath.append(val)
        trainpath.append(train)

        fortrain = pd.DataFrame()
        fortrainid = pd.DataFrame()
        for id in train:
            fortrainid = clinicdata[clinicdata['UID'] == id]
            fortrain = pd.concat([fortrain, fortrainid])

        tmpuid = fortrain['UID'].tolist()
        tmptarget = fortrain['sure'].tolist()
        zscore = ZS.fit(fortrain.iloc[:, 1:-1])
        fortraindata = zscore.transform(fortrain.iloc[:, 1:-1])
        fortrain = pd.DataFrame(fortraindata, columns=c_namedata)
        fortrain.insert(0, 'UID', tmpuid)
        fortrain.insert(33, 'sure', tmptarget)
        forval = pd.DataFrame()
        forvalid = pd.DataFrame()
        for id in val:
            forvalid = clinicdata[clinicdata['UID'] == id]
            forval = pd.concat([forval, forvalid])
        tmpuid = forval['UID'].tolist()
        tmptarget = forval['sure'].tolist()
        forvaldata = zscore.transform(forval.iloc[:, 1:-1])
        forval = pd.DataFrame(forvaldata, columns=c_namedata)
        forval.insert(0, 'UID', tmpuid)
        forval.insert(33, 'sure', tmptarget)
        clinicdata_train.append(fortrain)
        clinicdata_val.append(forval)
    return clinicdata_val, clinicdata_train



def main():
    k = 5
    num_classes = 3



    train_accuracy_all = []
    val_accuracy_all = []

    clinicdata_val, clinicdata_train = cross_val(k)

    for fold in range(k):
        valx = []
        valy = []
        trainx = []
        trainy = []

        trainscore_list = []
        trainlabel_list = []
        valscore_list = []
        vallabel_list = []
        clinicfile = '..'  # ID features target
        clinicdata = pd.read_excel(clinicfile)
        for val_id in clinicdata_val[fold]['UID'].tolist():
            valx.append(val_id)
            tmp = clinicdata[clinicdata['UID'] == val_id]
            if tmp['sure'].values == 0:
                label = 0
            elif tmp['sure'].values == 1:
                label = 1
            else:
                label = 2
            valy.append(label)
        clinicfeature_val, label_val = datasetcv_val(valx, valy, clinicdata_val, fold)
        for train_id in clinicdata_train[fold]['UID'].tolist():
            trainx.append(train_id)
            tmp = clinicdata[clinicdata['UID'] == train_id]
            if tmp['sure'].values == 0:
                label = 0
            elif tmp['sure'].values == 1:
                label = 1
            else:
                label = 2
            trainy.append(label)

        clinicfeature_train, label_train = datasetcv_train(trainx, trainy, clinicdata_train, fold)
        model_svm = OneVsRestClassifier(svm.SVC(C=1, probability=True,kernel='rbf'))

        model_svm.fit(clinicfeature_train,label_train)


        valpredict = model_svm.predict(clinicfeature_val)
        valpredictprob = model_svm.predict_proba(clinicfeature_val)

        total = 0
        correct = 0
        total = len(label_val)
        correct = np.equal(valpredict, label_val).sum()
        val_accuracy = correct / total
        print("\n accuracy on valid data: %.2f%%\n" % (100 * val_accuracy))


        vallabel_onehot = np.eye(num_classes, dtype=np.uint8)[label_val]
        valscore_array = valpredictprob


        fpr_dict = dict()
        tpr_dict = dict()
        roc_auc_dict = dict()
        se = dict()
        lowerb = dict()
        upperb = dict()
        p = dict()

        class_names = np.array(['MAP', 'MSAP', 'SAP'])
        y_true =  valpredict

        plot_confusion_matrix(label_val,  valpredict, classes=class_names, normalize=False)

        for i in range(num_classes):
            alpha = 0.05
            fpr_dict[i], tpr_dict[i], _ = roc_curve( vallabel_onehot[:, i],  valscore_array[:, i])
            roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
            n1 = np.sum( vallabel_onehot[:, i] == 1)

            n2 =  vallabel_onehot.shape[0] - n1

            q1 = roc_auc_dict[i] / (2 - roc_auc_dict[i])

            q2 = (2 * roc_auc_dict[i] ** 2) / (1 + roc_auc_dict[i])

            se[i] = np.sqrt((roc_auc_dict[i] * (1 - roc_auc_dict[i]) + (n1 - 1) * (q1 - roc_auc_dict[i] ** 2) + (
                    n2 - 1) * (q2 - roc_auc_dict[i] ** 2)) / (n1 * n2))

            confidence_level = 1 - alpha
            z = (roc_auc_dict[i] - 0.5) / se[i]
            p[i] = 2 * (1 - norm.cdf(z, loc=0, scale=1))

            z_lower, z_upper = norm.interval(confidence_level)
            lowerb[i], upperb[i] = roc_auc_dict[i] + z_lower * se[i], min(roc_auc_dict[i] + z_upper * se[i], 1)
            print(p[i])

        # micro
        fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve( vallabel_onehot.ravel(),  valscore_array.ravel())
        roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])

        # macro
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(num_classes)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(num_classes):
            mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
        # Finally average it and compute AUC
        mean_tpr /= num_classes
        fpr_dict["macro"] = all_fpr
        tpr_dict["macro"] = mean_tpr
        roc_auc_dict["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])


        plt.figure()
        lw = 2

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

        for i, color in zip(range(num_classes), colors):
            plt.plot(fpr_dict[i], tpr_dict[i], color=color, lw=lw,
                     label='ROC curve of {0} (area = {1:0.2f}(95%CI:{2:0.2f}-{3:0.2f}))'.format(class_names[i],
                                                                                                      roc_auc_dict[i],
                                                                                                      lowerb[i],
                                                                                                      upperb[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.savefig('val' + str(fold + 1) + 'foldroc.jpg')
        plt.show()



if __name__ == "__main__":
    main()