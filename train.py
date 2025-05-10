import os.path as osp
import random

import torch
from torch.utils.data import DataLoader

import settings
from models_noise import *
from dataset import *
from triple_loss import *
from sklearn.metrics import accuracy_score,f1_score,classification_report
import numpy as np
import time
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import torch.nn.functional as F
from sklearn import metrics

torch.manual_seed(200)
torch.cuda.manual_seed_all(200)

class Session:
    def __init__(self):
        self.logger = settings.logger
        torch.cuda.set_device(settings.GPU_ID)
        print(torch.cuda.is_available())

        # Weibo数据
        if(settings.DATASET== 'Weibo'):
            # 微博
            self.test_images = np.load(
            r"F:\A多模态项目\CCAH-main未改2\CCAH-main未改\CCAH\data\weibo_2\weibo_clip_test_image.npy",
            allow_pickle=True)
            self.test_tags = np.load(r"F:\A多模态项目\CCAH-main未改2\CCAH-main未改\CCAH\data\weibo_2\weibo_test_text.npy",
                            allow_pickle=True)
            self.test_labels = np.load(r"F:\A多模态项目\CCAH-main未改2\CCAH-main未改\CCAH\data\weibo_2\weibo_test_label.npy",
                              allow_pickle=True)
            self.test_event = np.load(r"F:\A多模态项目\CCAH-main未改2\CCAH-main未改\CCAH\data\weibo_2\weibo_test_event.npy",
                             allow_pickle=True)

        self.model1 = teacher_model()
        self.model2 = teacher_model2()
        self.model3 = student_model()
        self.kl_div = torch.nn.KLDivLoss()
        self.softmax = torch.nn.Softmax(dim=1)
        self.opt_model1 = torch.optim.SGD(self.model1.parameters(), lr=settings.LR_IMG, momentum=settings.MOMENTUM,
                                         weight_decay=settings.WEIGHT_DECAY, nesterov=True)
        self.opt_model2 = torch.optim.SGD(self.model2.parameters(), lr=settings.LR_IMG, momentum=settings.MOMENTUM,
                                         weight_decay=settings.WEIGHT_DECAY, nesterov=True)
        self.opt_model3 = torch.optim.SGD(self.model3.parameters(), lr=settings.LR_IMG, momentum=settings.MOMENTUM,
                                         weight_decay=settings.WEIGHT_DECAY, nesterov=True)
        self.best = 0

    def train(self, epoch,train_images,train_tags,train_labels,train_events,
                 un_train_images,un_train_tags,un_train_labels,un_train_events):
        self.train_images = train_images
        self.train_tags = train_tags
        self.train_labels = train_labels
        self.train_event = train_events

        self.un_train_images = un_train_images
        self.un_train_tags = un_train_tags
        self.un_train_labels = un_train_labels
        self.un_train_event = un_train_events
        self.un_train_event2 = np.ones(len(un_train_labels))

        #如果所有数据标注完，不用进行拼接
        if(len(self.un_train_labels)>0):
            for i in range(len(self.un_train_labels)):
                self.un_train_event2[i] = 3
            self.train_images_T = np.concatenate((self.train_images,self.un_train_images),axis=0)
            self.train_tags_T = np.concatenate((self.train_tags,self.un_train_tags),axis=0)
            self.train_labels_T = np.concatenate((self.train_labels, self.un_train_labels), axis=0)
            self.train_events_T = np.concatenate((self.train_event, self.un_train_event2), axis=0)
        else:
            self.train_images_T = self.train_images
            self.train_tags_T = self.train_tags
            self.train_labels_T = self.train_labels
            self.train_events_T = self.train_event

        self.train_dataset_T = Dataset1(self.train_images_T, self.train_tags_T, self.train_labels_T, self.train_events_T)
        self.train_dataset = Dataset1(self.train_images, self.train_tags, self.train_labels, self.train_event)
        self.un_train_dataset = Dataset1(self.un_train_images, self.un_train_tags, self.un_train_labels,
                                         self.un_train_event)
        # self.pre_train_dataset = Dataset1(self.pre_train_images, self.pre_train_tags, self.pre_train_labels,
        #                                  self.pre_train_event)
        self.test_dataset = Dataset1(self.test_images, self.test_tags, self.test_labels, self.test_event)

        self.train_loader_T = DataLoader(dataset=self.train_dataset_T,
                                       batch_size=settings.BATCH_SIZE,
                                       shuffle=True,
                                       num_workers=settings.NUM_WORKERS, drop_last=True)

        self.train_loader = DataLoader(dataset=self.train_dataset,
                                       batch_size=settings.BATCH_SIZE,
                                       shuffle=True,
                                       num_workers=settings.NUM_WORKERS,drop_last=True)


        self.un_train_loader = DataLoader(dataset=self.un_train_dataset,
                                          batch_size=settings.BATCH_SIZE,
                                          shuffle=False,
                                          num_workers=settings.NUM_WORKERS)

        self.test_loader = DataLoader(dataset=self.test_dataset,
                                      batch_size=settings.BATCH_SIZE,
                                      shuffle=False,
                                      num_workers=settings.NUM_WORKERS)
        self.model1.cuda().train()
        self.model2.cuda().train()
        self.model3.cuda().train()
        self.logger.info('Epoch [%d/%d]' % (
            epoch + 1, settings.NUM_EPOCH))
        print("训练数据量：",len(self.train_labels))
        # path_state_dict = "F:\A半监督\SDSA干净版\checkpoint\Weibo_64_bit_best_epoch.pth"
        # state = torch.load(path_state_dict)
        # self.model1.load_state_dict(state)
        a = 0
        for idx, (img, txt, events, labels) in enumerate(self.train_loader):
            txt = torch.FloatTensor(txt.numpy()).cuda()
            """图像为Tensor类型"""
            img = torch.FloatTensor(img.numpy()).cuda()
            a = a + len(labels)
            # self.opt_model.zero_grad()
            self.opt_model3.zero_grad()
            loss_fn = nn.CrossEntropyLoss()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            loss,pred,fusion_teacher,code_I1,code_T1 = self.model1(img,txt,labels,events,'train')
            loss2,pred2,fusion_teacher2,code_I2,code_T2 = self.model2(img,txt,labels,events,'train')
            loss3, pred3, fusion_student,code_I3,code_T3 = self.model3(img, txt, labels, events, 'train')

            # 特征蒸馏
            F_I1 = F.normalize(code_I1)
            F_T1 = F.normalize(code_T1)
            F_I2 = F.normalize(code_I2)
            F_T2 = F.normalize(code_T2)
            F_I3 = F.normalize(code_I3)
            F_T3 = F.normalize(code_T3)
            # 图像特征蒸馏损失
            feature_I_loss1 = F.mse_loss(F_I3,F_I1)
            feature_I_loss2 = F.mse_loss(F_I3, F_I2)
            # 文本特征蒸馏损失
            feature_T_loss1 = F.mse_loss(F_T3,F_T1)
            feature_T_loss2 = F.mse_loss(F_T3, F_T2)

            feature_loss = feature_I_loss1 + feature_I_loss2 + feature_T_loss1 + feature_T_loss2

            # 结构蒸馏
            # 图像邻接结构
            S_I1 = F_I1.mm(F_I1.t())
            S_I2 = F_I2.mm(F_I2.t())
            S_I3 = F_I3.mm(F_I3.t())
            # 文本邻接结构
            S_T1 = F_T1.mm(F_T1.t())
            S_T2 = F_T2.mm(F_T2.t())
            S_T3 = F_T3.mm(F_T3.t())

            # 图像结构蒸馏损失
            S_I_loss1 = F.mse_loss(S_I3, S_I1)
            S_I_loss2 = F.mse_loss(S_I3, S_I2)
            # 文本结构蒸馏损失
            S_T_loss1 = F.mse_loss(S_T3, S_T1)
            S_T_loss2 = F.mse_loss(S_T3, S_T2)

            S_loss = S_I_loss1 + S_I_loss2 + S_T_loss1 + S_T_loss2
            # 结构蒸馏

            loss3 = 1*loss3 + 0.5*(S_loss + feature_loss)
            loss3.backward()
            # loss.backward()
            # self.opt_model.step()
            self.opt_model3.step()

            if (idx + 1) % (len(self.train_dataset) // settings.BATCH_SIZE / settings.EPOCH_INTERVAL) == 0:
                self.logger.info(
                    'Epoch [%d/%d], Iter [%d/%d] '
                    'Total Loss: %.4f'
                    'Total Loss2: %.4f'
                    % (
                        epoch + 1, settings.NUM_EPOCH, idx + 1,
                        len(self.train_dataset) // settings.BATCH_SIZE,
                        loss3.item(),
                    loss3.item())
                )
            # print("训练数据量：",a)

    def train_eval(self, step, last=False):
        self.model1.eval().cuda()
        self.model2.eval().cuda()
        re_L = list([])
        test_label_pred1 = list([])
        t0 = time.perf_counter()
        labeled_index = ([])
        labeled_label = ([])
        num = 0
        for idx, (img, txt, event,labels) in enumerate(self.un_train_loader):
            txt_feature = txt
            txt = torch.FloatTensor(txt_feature.numpy()).cuda()
            img = torch.FloatTensor(img.numpy()).cuda()
            loss,test_label_pred,fusion,code_I,code_T = self.model1(img,txt,labels,event,'train_eval')
            loss2, test_label_pred2, fusion2,code_I,code_T = self.model2(img, txt, labels, event, 'train_eval')
            # 将logit转换为标签
            pre_label_detection = test_label_pred.argmax(1)
            pre_label_detection2 = test_label_pred2.argmax(1)
            prob = F.softmax(test_label_pred, dim=1)
            prob2 = F.softmax(test_label_pred2, dim=1)

            # 多教师模型判断高质量标签
            for i in range(len(labels)):
                # 取2个教师第一位logit
                a = pre_label_detection[i]
                b = pre_label_detection2[i]

                if (a==b):
                    labeled_index.append(event[i].cpu().numpy())
                    labeled_label.append(pre_label_detection[i].cpu().numpy())
                    num = num + 1


        labeled_index = np.array(labeled_index)
        labeled_label = np.array(labeled_label)
        print(num)


        return labeled_index,labeled_label


    def eval(self,step,symbol,  last=False):
        self.model3.eval().cuda()
        re_L = list([])
        test_label_pred1 = list([])
        t0 = time.perf_counter()
        re_BI = list([])

        for idx, (img, txt, event,labels) in enumerate(self.test_loader):
            txt_feature = txt
            txt = torch.FloatTensor(txt_feature.numpy()).cuda()
            img = torch.FloatTensor(img.numpy()).cuda()
            loss,test_label_pred,fusion,code_I,code_T = self.model3(img,txt,labels,event,'test')
            pre_label_detection = test_label_pred.argmax(1)
            test_label_pred1.extend(pre_label_detection.cpu().data.numpy())
            re_L.extend(labels.cpu().data.numpy())
            re_BI.extend(fusion.cpu().data.numpy())


        re_L = np.array(re_L)
        re_BI = np.array(re_BI)
        qu_L = self.test_dataset.train_labels
        test_label_pred = np.array(test_label_pred1)
        test_accuracy = accuracy_score(re_L, test_label_pred)
        classreport = classification_report(re_L, test_label_pred, digits=3)
        confusion_matrix = metrics.confusion_matrix(re_L,test_label_pred)       # 混淆矩阵（注意与上面示例的混淆矩阵的图位置并不一一对应）
        tn, fp, fn, tp = metrics.confusion_matrix(re_L,test_label_pred).ravel() # 混淆矩阵各值
        print(confusion_matrix)
        print(tn,fp,fn,tp)
        # if(symbol==1):
        #     tsne(re_BI, qu_L)
        print(test_accuracy)
        self.logger.info('ACC %.4f' % test_accuracy )
        t1 = time.perf_counter()  # 放在测试过程前后

        test_time = 'time:{:.6f}'.format((t1 - t0) / 10000)  # 除以的是测试样本
        print(test_time)
        print(classreport)
        if test_accuracy > settings.best:
            settings.best = test_accuracy
            self.save_checkpoints_student(step=step, best=True)
            settings.best = test_accuracy
            self.logger.info("#########is best:%.3f #########" % settings.best)
        print(settings.best)

        return test_accuracy

    def reload_data(self,step):
        self.train_images = np.load(
                r"F:\A多模态项目\CCAH-main未改2\CCAH-main未改\CCAH\data\weibo_2\Weibo_train_image_Resnet34.npy",
                allow_pickle=True)
        self.train_tags = np.load(r"F:\A多模态项目\CCAH-main未改2\CCAH-main未改\CCAH\data\weibo_2\weibo_train_text.npy",
                                 allow_pickle=True)
        self.train_labels = np.load(r"F:\A多模态项目\CCAH-main未改2\CCAH-main未改\CCAH\data\weibo_2\weibo_train_label.npy",
                                   allow_pickle=True)
        self.train_event = np.load(r"F:\A多模态项目\CCAH-main未改2\CCAH-main未改\CCAH\data\weibo_2\weibo_train_event.npy",
                                  allow_pickle=True)


    def save_checkpoints(self, step, file_name='%s_%d_bit_latest.pth' % (settings.DATASET, settings.BATCH_SIZE),
                         best=False):
        if best:
            file_name = '%s_%d_bit_teacher_best_epoch.pth' % (settings.DATASET, settings.BATCH_SIZE)
        ckp_path = osp.join(settings.MODEL_DIR, file_name)
        obj = {
            'model': self.model1.state_dict(),
            'step': step,
        }
        torch.save(obj, ckp_path)
        self.logger.info('**********Save the trained model successfully.**********')

    def load_checkpoints(self, file_name='%s_%d_bit_teacher_best_epoch.pth' % (settings.DATASET, settings.BATCH_SIZE)):
        ckp_path = osp.join(settings.MODEL_DIR, file_name)
        try:
            obj = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
            self.logger.info('**************** Load checkpoint %s ****************' % ckp_path)
        except IOError:
            self.logger.error('********** No checkpoint %s!*********' % ckp_path)
            return
        self.model1.load_state_dict(obj['model'])
        self.logger.info('********** The loaded model has been trained for epochs.*********')

    def save_checkpoints_teacher2(self, step, file_name='%s_%d_bit_latest.pth' % (settings.DATASET, settings.BATCH_SIZE),
                         best=False):
        if best:
            file_name = '%s_%d_bit_teacher2_best_epoch.pth' % (settings.DATASET, settings.BATCH_SIZE)
        ckp_path = osp.join(settings.MODEL_DIR, file_name)
        obj = {
            'model': self.model2.state_dict(),
            'step': step,
        }
        torch.save(obj, ckp_path)
        self.logger.info('**********Save the trained model successfully.**********')

    def load_checkpoints_teacher2(self, file_name='%s_%d_bit_teacher2_best_epoch.pth' % (settings.DATASET, settings.BATCH_SIZE)):
        ckp_path = osp.join(settings.MODEL_DIR, file_name)
        try:
            obj = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
            self.logger.info('**************** Load checkpoint %s ****************' % ckp_path)
        except IOError:
            self.logger.error('********** No checkpoint %s!*********' % ckp_path)
            return
        self.model2.load_state_dict(obj['model'])
        self.logger.info('********** The loaded model has been trained for epochs.*********')

    def save_checkpoints_student(self, step, file_name='%s_%d_student_bit_latest.pth' % (settings.DATASET, settings.BATCH_SIZE),
                         best=False):
        if best:
            file_name = '%s_%d_student_bit_best_epoch.pth' % (settings.DATASET, settings.BATCH_SIZE)
        ckp_path = osp.join(settings.MODEL_DIR, file_name)
        obj = {
            'model': self.model3.state_dict(),
            'step': step,
        }
        torch.save(obj, ckp_path)
        self.logger.info('**********Save the trained model successfully.**********')

    def load_checkpoints_student(self, file_name='%s_%d_student_bit_best_epoch.pth' % (settings.DATASET, settings.BATCH_SIZE)):
        ckp_path = osp.join(settings.MODEL_DIR, file_name)
        try:
            obj = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
            self.logger.info('**************** Load checkpoint %s ****************' % ckp_path)
        except IOError:
            self.logger.error('********** No checkpoint %s!*********' % ckp_path)
            return
        self.model3.load_state_dict(obj['model'])
        self.logger.info('********** The loaded model has been trained for epochs.*********')

    def random_int_list(self,start, stop, length):
        start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
        length = int(abs(length)) if length else 0
        random_list = []
        for i in range(stop):
            n = 0
            a = random.randint(start, stop)
            for x in range(len(random_list)):
                m = random_list[x]
                if(a==m):
                    n = n+1
            if(n==0):
                random_list.append(a)
            if(len(random_list)==length):
                break
        return random_list

def tsne(input, labels):
    # input = input.numpy()
    # labels = labels.numpy()
    tsne = TSNE(n_components=2, init='pca', random_state=0, n_iter=5000, perplexity=15)
    result = tsne.fit_transform(input)
    type1_x = []
    type1_y = []
    type2_x = []
    type2_y = []
    label = labels
    for i in range(result.shape[0]):
        if label[i] == 0:
            type1_x.append(result[i][0])
            type1_y.append(result[i][1])
        if label[i] == 1:
            type2_x.append(result[i][0])
            type2_y.append(result[i][1])

    #plt.title("TSNE X")
    #plt.xlim(xmin=-100, xmax=100)
    #plt.ylim(ymin=-100, ymax=150)
    type1 = plt.scatter(type1_x, type1_y, s=10, c='r',marker='<')
    type2 = plt.scatter(type2_x, type2_y, s=10, c='g',marker='<')
    plt.legend((type1, type2), ('Fake', 'Real'))
    plt.savefig('C:/Users/yue/Desktop/半监督方法/半监督实验/Pheme_TSNE.png',dpi=300)
    # plt.show()
def plot(input_list):
    adopt = np.load('./fakediitAcc.npy', allow_pickle=True)
    # consistence = np.load('./labeled_acc_一致性正则化_weibo.npy')
    all =  np.load('./fakediitAcc-js.npy', allow_pickle=True)
    adopt[0] = 0
    all[0] = 0
    adopt = adopt[0:100]
    # consistence = consistence[0:100]
    all = all[0:100]
    a = input_list
    b = list(range(len(adopt)))

    plt.figure(dpi=150)
    plt.plot(b, adopt, label='TKDN',  markersize=4, )
    # plt.plot(b, consistence, label='QMFND2',  markersize=4, )
    plt.plot(b, all, label='TKDN-js', markersize=4, )
    plt.ylim((0, 0.95))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(False)
    plt.savefig("./过拟合_Fakeddit.png",dpi=150)
    plt.show()

def main():
    for x in range(1):
        # 微博
        train_images = np.load(
            r"F:\A多模态项目\CCAH-main未改2\CCAH-main未改\CCAH\data\weibo_2\weibo_clip_train_image.npy",
            allow_pickle=True)
        train_tags = np.load(r"F:\A多模态项目\CCAH-main未改2\CCAH-main未改\CCAH\data\weibo_2\weibo_train_text.npy",
                                  allow_pickle=True)
        train_labels = np.load(
            r"F:\A多模态项目\CCAH-main未改2\CCAH-main未改\CCAH\data\weibo_2\weibo_train_label.npy",
            allow_pickle=True)

        # index = random.sample(range(0,29999),18000)
        # np.save("./fakedditindex20.6.npy",index)
        index = np.load("./phemeindex20.2.npy")
        index.sort()
        labeled_index = np.zeros(283, dtype=int)
        unlabeled_index = np.zeros(1129, dtype=int)
        a = 0
        b = 0
        symbol = 0
        # 获得未标记数据下标
        for i in range(1412):
            if (symbol == 283):
                symbol = 282
            if (i == index[symbol]):
                labeled_index[a] = i
                symbol = symbol + 1
                a = a + 1

            else:
                unlabeled_index[b] = i
                b = b + 1


        #标注数据
        labeled_train_images = [train_images[i] for i in labeled_index]
        labeled_train_tags = [train_tags[i] for i in labeled_index]
        labeled_train_labels = [train_labels[i] for i in labeled_index]
        labeled_train_images = np.array(labeled_train_images)
        labeled_train_tags = np.array(labeled_train_tags)
        labeled_train_labels = np.array(labeled_train_labels)
        labeled_symbol = np.zeros(len(labeled_train_labels),dtype=int)

        #未标注数据
        unlabeled_train_images = [train_images[i] for i in unlabeled_index]
        unlabeled_train_tags = [train_tags[i] for i in unlabeled_index]
        unlabeled_train_labels = [train_labels[i] for i in unlabeled_index]
        unlabeled_train_images = np.array(unlabeled_train_images)
        unlabeled_train_tags = np.array(unlabeled_train_tags)
        unlabeled_train_labels = np.array(unlabeled_train_labels)



        sess = Session()
        # 加载教师模型参数
        sess.load_checkpoints()
        sess.load_checkpoints_teacher2()

        test_Acc_list = list([])

        for epoch in range(settings.NUM_EPOCH):
            # train the Model
            # sess = Session(labeled_train_images, labeled_train_tags, labeled_train_labels,labeled_symbol,
            #                unlabeled_train_images, unlabeled_train_tags, unlabeled_train_labels, unlabeled_index)
            sess.train(epoch,labeled_train_images, labeled_train_tags, labeled_train_labels, labeled_symbol,
                       unlabeled_train_images, unlabeled_train_tags, unlabeled_train_labels, unlabeled_index)
            pre_labeled_index,pre_labeled_label= sess.train_eval(step=epoch + 1)
            # eval the Model
            if (epoch + 1) % 1 == 0:
                test_Acc = sess.eval(step=epoch + 1,symbol=0)
                test_Acc_list.append(test_Acc)
                
            # sess.reload_data(step=1)

            if(len(unlabeled_index)>0):
                #删除已标注数据下标
                unlabeled_index =unlabeled_index.tolist()
                for i in pre_labeled_index:
                    unlabeled_index.remove(i)

                unlabeled_index = np.array(unlabeled_index)
                # 加入已标注数据下标
                labeled_index = np.append(labeled_index, pre_labeled_index)
                # labeled_index.astype(np.int16)

                # 重载已标注数据

                labeled_train_labels = np.append(labeled_train_labels, pre_labeled_label)
                labeled_train_images = [train_images[i.astype(np.int16)] for i in labeled_index]
                labeled_train_tags = [train_tags[i.astype(np.int16)] for i in labeled_index]
                labeled_train_images = np.array(labeled_train_images)
                labeled_train_tags = np.array(labeled_train_tags)
                labeled_symbol2 = np.ones(len(pre_labeled_label),dtype=int)
                labeled_symbol = np.append(labeled_symbol, labeled_symbol2)

                # 重载未标注数据
                unlabeled_train_images = [train_images[i] for i in unlabeled_index]
                unlabeled_train_tags = [train_tags[i] for i in unlabeled_index]
                unlabeled_train_labels = [train_labels[i] for i in unlabeled_index]
                unlabeled_train_images = np.array(unlabeled_train_images)
                unlabeled_train_tags = np.array(unlabeled_train_tags)
                unlabeled_train_labels = np.array(unlabeled_train_labels)

        np.save("./fakediitAcc.npy",test_Acc_list)
        sess.logger.info('---------------------------Test------------------------')
        sess.load_checkpoints_student()
        sess.eval(step=settings.BATCH_SIZE,symbol=1)


if __name__ == '__main__':
    main()
