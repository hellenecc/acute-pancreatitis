import torch as t
from torch.utils import data
import os
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score,roc_auc_score
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from mymodel_vgg_two_class import AttnVGG
from utilities import *
from PIL import Image
import numpy as np
import torchvision
import torchvision.utils as utils
import torchvision.transforms as transforms
from torch.utils.data import random_split
import random
import shutil
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
from scipy.stats import norm
import pandas as pd
from pytorchtools import EarlyStopping

import os
gpuid = '0,1,2,3,4,5,6,7,8,9'
os.environ['CUDA_VISIBLE_DEVICES'] = gpuid

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

class datasetcv(data.Dataset):
    def __init__(self,x,y,transforms=None):
        self.imgs=x
        self.labels=y
        self.transforms = transforms
    def __getitem__(self, index):
        img_path=self.imgs[index]
        label=self.labels[index]
        data=Image.open(img_path)
        if self.transforms:
            data=self.transforms(data)
        return data,label
    def __len__(self):
        return len(self.imgs)

transform_train = transforms.Compose([
        transforms.ToTensor(),
])

transform_val = transforms.Compose([
        transforms.ToTensor(),
])

def cross_val(root, k):
    valpath=[]
    trainpath=[]

    imgs= os.listdir(root)
    random.shuffle(imgs)
    imglevel1=[img for img in imgs if   'level1' in img]
    imglevel2=[img for img in imgs if   'level2' in img]
    for i in range(k):
        val = []
        train = []
        imgs1 = [os.path.join(root, img) for img in imglevel1]
        nval1 = len(imgs1) // k
        val.extend(imgs1[nval1 * i:nval1 * (i + 1)])
        train.extend(imgs1[:nval1 * i]+imgs1[(i + 1) * nval1:])
        imgs2 = [os.path.join(root, img) for img in imglevel2]
        nval2 = len(imgs2) // k
        val.extend(imgs2[nval2 * i:nval2 * (i + 1)])
        train.extend(imgs2[:nval2 * i] + imgs2[(i + 1) * nval2:])
        valpath.append(val)
        trainpath.append(train)
    return valpath,trainpath



def main():
    k = 5
    patience = 20
    root = ''
    valpath, trainpath = cross_val(root, k)
    normalize_attn = True
    num_classes = 2
    im_size = 512
    num_aug = 1
    nrow = 1
    lr=0.01
    epochs=300
    logs='../logs'
    batch_size=64
    log_images=True
    writer = SummaryWriter(logs)

    train_accuracy_all = []
    val_accuracy_all = []


    valpath, trainpath = cross_val(root, k)

    for fold in range(k):
        valx=[]
        valy=[]
        trainx =[]
        trainy =[]
        early_stopping = EarlyStopping(patience, verbose=True)



        for valfile in valpath[fold]:
            valx.append(valfile)
            if 'level1' in valfile.split('/')[-1]:
                label = 0
            elif 'level2' in valfile.split('/')[-1]:
                label = 1
            valy.append(label)
        print('valdataset%d_size :%d' %(fold,len(valx)))
        val_dataset=datasetcv(valx,valy,transform_val)

        for trainfile in trainpath[fold]:
            trainx.append(trainfile)
            if 'level1' in trainfile.split('/')[-1]:
                label = 0
            elif 'level2' in trainfile.split('/')[-1]:
                label = 1
            trainy.append(label)
        print('traindataset%d_size :%d' % (fold,len(trainx)))

        train_dataset=datasetcv(trainx,trainy,transform_train)

        trainloader = t.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        valloader = t.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        net = AttnVGG(im_size=im_size, num_classes=2, normalize_attn=normalize_attn, init='xavierUniform')

        criterion=nn.CrossEntropyLoss()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_ids = [0,1,2,3,4,5,6,7,8,9]
        model = nn.DataParallel(net, device_ids=device_ids).to(device)
        criterion.to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        lr_lambda = lambda epoch: np.power(0.5, int(epoch / 25))
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        step = 0
        running_avg_accuracy = 0
        trainscore_list = []
        trainlabel_list = []
        valscore_list = []
        vallabel_list = []


        for epoch in range(epochs):
            images_disp = []
            # adjust learning rate
            scheduler.step()
            writer.add_scalar('train/learning_rate_'+str(fold)+'fold', optimizer.param_groups[0]['lr'], epoch)
            print("\n%d fold epoch %d learning rate %f\n" % (fold,epoch, optimizer.param_groups[0]['lr']))
            # run for one epoch
            for aug in range(num_aug):
                for i, data in enumerate(trainloader, 0):
                    # warm up
                    model.train()
                    model.zero_grad()
                    optimizer.zero_grad()
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    if (aug == 0) and (i == 0):  # archive images in order to save to logs
                        images_disp.append(inputs[0:1,:, :, :])
                    # forward

                    pred, __, __, __ = model(inputs)



                    # backward
                    loss = criterion(pred, labels)
                    loss.backward()
                    optimizer.step()
                    # display results
                    if i % 10 == 0:
                        model.eval()
                        predtrain, __, __, __ = model(inputs)
                        predict = torch.argmax(predtrain, 1)
                        total = labels.size(0)
                        correct = torch.eq(predict, labels).sum().double().item()
                        train_accuracy = correct / total
                        running_avg_accuracy = 0.9 * running_avg_accuracy + 0.1 * train_accuracy
                        writer.add_scalar('train/loss_'+str(fold)+'fold', loss.item(), step)
                        writer.add_scalar('train/accuracy_'+str(fold)+'fold', train_accuracy, step)
                        writer.add_scalar('train/running_avg_accuracy_'+str(fold)+'fold', running_avg_accuracy, step)
                        print("[fold %d][epoch %d][aug %d/%d][%d/%d] loss %.4f accuracy %.2f%% running avg accuracy %.2f%%"
                              % (fold,epoch, aug, num_aug - 1, i, len(trainloader) - 1, loss.item(), (100 * train_accuracy),
                                 (100 * running_avg_accuracy)))
                    step += 1
            # the end of each epoch: test & log


            print('\none epoch done, saving records ...\n')
            torch.save(model.state_dict(), os.path.join(logs, 'net.pth'))
            weights_all = model.state_dict()['module.classify.weight']



            model.eval()
            total = 0
            correct = 0
            loss_val = 0
            with torch.no_grad():
                # log scalars
                for i, data in enumerate(valloader, 0):
                    images_val, labels_val = data
                    images_val, labels_val = images_val.to(device), labels_val.to(device)
                    if i == 0:  # archive images in order to save to logs
                        images_disp.append(images_val[0:1, :, :, :])
                    pred_test, __, __, __ = model(images_val)
                    loss_val += criterion(pred_test, labels_val) * len(labels_val)
                    predict = torch.argmax(pred_test, 1)
                    total += labels_val.size(0)
                    correct += torch.eq(predict, labels_val).sum().double().item()
                val_accuracy=correct / total
                loss_val = loss_val / total
                writer.add_scalar('val/accuracy_'+str(fold), val_accuracy, epoch)
                print("\n[epoch %d] accuracy on test data: %.2f%%\n" % (epoch, 100 * val_accuracy))
                # log images
                if log_images:
                    I_train = utils.make_grid(images_disp[0], nrow, normalize=True, scale_each=True, padding=0)
                    writer.add_image('train/image_'+str(fold), I_train, epoch)
                    I_test = utils.make_grid(images_disp[1], nrow, normalize=True, scale_each=True, padding=0)
                    writer.add_image('val/image_'+str(fold), I_test, epoch)

                if log_images:
                    print('\nlog attention maps ...\n')

                    min_up_factor = 16

                    if normalize_attn:
                        vis_fun = visualize_attn_softmax
                    else:
                        vis_fun = visualize_attn_sigmoid

                    # training data
                    __, c1, c2, c3 = model(images_disp[0])
                    if c1 is not None:
                        attn1 = vis_fun(I_train, c1, up_factor=min_up_factor, nrow=nrow)
                        writer.add_image('train/attention_map_1_'+str(fold), attn1, epoch)
                    if c2 is not None:
                        attn2 = vis_fun(I_train, c2, up_factor=min_up_factor * 2, nrow=nrow)
                        writer.add_image('train/attention_map_2_'+str(fold), attn2, epoch)

                    if c3 is not None:
                        attn3 = vis_fun(I_train, c3, up_factor=min_up_factor * 4, nrow=nrow)
                        writer.add_image('train/attention_map_3_'+str(fold), attn3, epoch)
                    # test data
                    __, c1, c2, c3 = model(images_disp[1])
                    if c1 is not None:
                        attn1 = vis_fun(I_test, c1, up_factor=min_up_factor, nrow=nrow)
                        writer.add_image('val/attention_map_1_'+str(fold), attn1, epoch)
                    if c2 is not None:
                        attn2 = vis_fun(I_test, c2, up_factor=min_up_factor * 2, nrow=nrow)
                        writer.add_image('val/attention_map_2_'+str(fold), attn2, epoch)
                    if c3 is not None:
                        attn3 = vis_fun(I_test, c3, up_factor=min_up_factor * 4, nrow=nrow)
                        writer.add_image('val/attention_map_3_'+str(fold), attn3, epoch)

                early_stopping(val_accuracy, model)
                if early_stopping.early_stop:
                    print("Early stopping")

                    break


        train_accuracy_all.append(train_accuracy)
        val_accuracy_all.append(val_accuracy)




        #end of all epoch
        model.load_state_dict(torch.load('checkpoint.pt'))
        for i, data in enumerate(valloader):
            inputsfold_val, labelsfold_val = data
            inputsfold_val, labelsfold_val = inputsfold_val.to(device), labelsfold_val.to(device)
            model.eval()


            predfold_val, __, __, __ = model(inputsfold_val)

            score_tmp = predfold_val

            valscore_list.extend(score_tmp.detach().cpu().numpy())
            vallabel_list.extend(labelsfold_val.cpu().numpy())
        valscore_array = np.array(valscore_list)
        y_pred = np.argmax(valscore_array, 1)
        valscore_array = softmax(valscore_array)


        vallabel_tensor = torch.tensor(vallabel_list)

        vallabel_tensor = vallabel_tensor.reshape((vallabel_tensor.shape[0], 1))



        vallabel_onehot = torch.zeros(vallabel_tensor.shape[0], num_classes)
        vallabel_onehot.scatter_(dim=1, index=vallabel_tensor, value=1)
        vallabel_onehot = np.array(vallabel_onehot)



        fpr_dict = dict()
        tpr_dict = dict()
        roc_auc_dict = dict()
        se = dict()
        lowerb = dict()
        upperb = dict()
        p = dict()


        class_names = np.array(['MAP', 'MSAP'])
        y_true = vallabel_list
        plot_confusion_matrix(y_true, y_pred, classes=class_names, normalize=False)

        for i in range(num_classes):
            alpha = 0.05
            fpr_dict[i], tpr_dict[i], _ = roc_curve(vallabel_onehot[:, i], valscore_array[:, i])
            roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
            n1 = np.sum(vallabel_onehot[:, i] == 1)

            n2 = vallabel_onehot.shape[0] - n1

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
        fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(vallabel_onehot.ravel(), valscore_array.ravel())
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
                     label='ROC curve of  {0} (area = {1:0.2f}(95%CI:{2:0.2f}-{3:0.2f}))'
                           ''.format(class_names[i], roc_auc_dict[i], lowerb[i], upperb[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.savefig('val' + str(fold + 1) + 'foldroc.jpg')
        plt.show()




    print(train_accuracy_all,val_accuracy_all)


if __name__ == "__main__":
    main()