import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import settings
from utlis import  ChannelCompress, to_edge
from torch.nn import init

class fcModal(nn.Module):
    def __init__(self):
        super(fcModal, self).__init__()
        self.fc_encode = nn.Sequential(
            nn.Linear(settings.CODE_LEN, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
            nn.Softmax(dim=1)
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(settings.CODE_LEN, 256),
            nn.LeakyReLU(),
            nn.Linear(256, settings.event),
            nn.Softmax(dim=1),
        )
        h_dim = 64
        self.classifier_corre = nn.Sequential(
        nn.Linear(settings.CODE_LEN, h_dim),
        nn.BatchNorm1d(h_dim),
        nn.ReLU(),
        nn.Linear(h_dim, h_dim),
        nn.BatchNorm1d(h_dim),
        nn.ReLU(),
        nn.Linear(h_dim, 2)
        )

    def forward(self, x,code_len):

        label_pred = self.classifier_corre(x)
        event_pred = self.domain_classifier(x)
        return label_pred, event_pred

class VIB_I(nn.Module):
    def __init__(self, in_ch=512, z_dim=256, num_class=2):
        super(VIB_I, self).__init__()
        self.in_ch = in_ch
        self.out_ch = z_dim * 2
        self.num_class = num_class
        self.bottleneck = ChannelCompress(in_ch=self.in_ch, out_ch=self.out_ch)
        # classifier of VIB, maybe modified later.
        classifier = []
        classifier += [nn.Linear(self.out_ch, self.out_ch // 2)]
        classifier += [nn.BatchNorm1d(self.out_ch // 2)]
        classifier += [nn.LeakyReLU(0.1)]
        classifier += [nn.Dropout(0.5)]
        classifier += [nn.Linear(self.out_ch // 2, self.num_class)]
        classifier = nn.Sequential(*classifier)
        self.classifier = classifier
        self.classifier.apply(weights_init_classifier)
        self.fc = nn.Linear(512, 512)


    def forward(self, v):
        v = self.fc(v)
        z_given_v = self.bottleneck(v)
        p_y_given_z = self.classifier(z_given_v)
        return  v,z_given_v

class VIB_T(nn.Module):
    def __init__(self, in_ch=512, z_dim=256, num_class=2):
        super(VIB_T, self).__init__()
        self.in_ch = in_ch
        self.out_ch = z_dim * 2
        self.num_class = num_class
        self.bottleneck = ChannelCompress(in_ch=self.in_ch, out_ch=self.out_ch)
            # classifier of VIB, maybe modified later.
        classifier = []
        classifier += [nn.Linear(self.out_ch, self.out_ch // 2)]
        classifier += [nn.BatchNorm1d(self.out_ch // 2)]
        classifier += [nn.LeakyReLU(0.1)]
        classifier += [nn.Dropout(0.5)]
        classifier += [nn.Linear(self.out_ch // 2, self.num_class)]
        classifier = nn.Sequential(*classifier)
        self.classifier = classifier
        self.classifier.apply(weights_init_classifier)
        self.fc = nn.Linear(768, 512)

    def forward(self, v):
        v = self.fc(v)
        z_given_v = self.bottleneck(v)
        p_y_given_z = self.classifier(z_given_v)
        return  v,z_given_v

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

class Attention_ast(nn.Module):
    def __init__(self, code_len):
        super(Attention_ast, self).__init__()
        # self.weight = torch.rand(100, 1)
        self.fc = nn.Linear(settings.CODE_LEN, 1)
    def forward(self, x):
        x2 = x
        x2 = self.fc(x2)
        x2 = torch.sigmoid(x2)
        return x2

class student_model(nn.Module):
    def __init__(self):
        super(student_model, self).__init__()
        self.fcModal = fcModal()
        self.attention = Attention_ast(code_len=settings.CODE_LEN)
        self.VIB_I = VIB_I()
        self.VIB_T = VIB_T()
        self.kl_div = torch.nn.KLDivLoss()
        self.softmax = torch.nn.Softmax(dim=1)
        self.fc_image = nn.Sequential(
            nn.Linear(settings.CODE_LEN, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
            nn.Softmax(dim=1)
        )
        self.fc_text = nn.Sequential(
            nn.Linear(settings.CODE_LEN, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, img,txt,label,event,step):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # add noise
        noise_std = 0.1
        img2 = img.cpu().numpy()
        txt2 = txt.cpu().numpy()
        img2 = img2 + noise_std * np.random.randn(*img2.shape)
        txt2 = txt2 + noise_std * np.random.randn(*txt2.shape)
        img3 = torch.from_numpy(img2)
        txt3 = torch.from_numpy(txt2)
        img3 = img3.to(torch.float32)
        txt3 = txt3.to(torch.float32)

        img, code_I = self.VIB_I(img)
        txt, code_T = self.VIB_T(txt)

        # 单个模态准确率
        pre_image = self.fc_image(code_I)
        pre_text = self.fc_text(code_T)
        
        att_A1 = self.attention(code_I)
        att_A2 = self.attention(code_T)
        att_A = torch.cat((att_A1, att_A2), 1)
        att_A = torch.softmax(att_A, dim=1)
        att_A1 = att_A[:, :1]
        att_A2 = att_A[:, 1:]
        F_A = torch.multiply(att_A1, code_I) + torch.multiply(att_A2, code_T)
        label_pred, event_pred = self.fcModal(F_A, settings.BATCH_SIZE)
        # label_pred = F.softmax(label_pred, dim=1)

        # # 生成噪声标签
        # txt3 = self.fc1(txt3.to(device))
        img3, code_I3 = self.VIB_I(img3.to(device))
        txt3, code_T3 = self.VIB_T(txt3.to(device))

        att_A13 = self.attention(code_I3)
        att_A23 = self.attention(code_T3)
        att_A3 = torch.cat((att_A13, att_A23), 1)
        att_A3 = torch.softmax(att_A3, dim=1)
        att_A13 = att_A3[:, :1]
        att_A23 = att_A3[:, 1:]
        F_A3 = torch.multiply(att_A13, code_I3) + torch.multiply(att_A23, code_T3)
        label_pred3, event_pred3 = self.fcModal(F_A3, settings.BATCH_SIZE)

        F_label_pred = F.normalize(label_pred)
        F_label_pred3 = F.normalize(label_pred3)
        #
        loss_mse = F.mse_loss(F_label_pred,F_label_pred3)
        # loss_mse = self.kl_div(input=self.softmax(F_label_pred3.detach() / 1),
        #             target=self.softmax(F_label_pred / 1))

        if (step == 'train'):
            vsd_loss = self.kl_div(input=self.softmax(img.detach() / 1),
                                   target=self.softmax(code_I / 1)) + \
                       self.kl_div(input=self.softmax(txt.detach() / 1),
                                   target=self.softmax(code_T / 1))

            loss_fn = nn.CrossEntropyLoss()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # 事实标签
            labeled = []
            labeled1 = torch.zeros(2)
            # 标注伪标签
            pre_label = []
            pre_label_image = []
            pre_label_text = []
            pre_label1 = torch.zeros(2)
            pre_label1_image = torch.zeros(2)
            pre_label1_text = torch.zeros(2)
            # 未标注数据
            labeled3 = []
            labeled_3 = torch.zeros(2)
            num = 0
            num2 = 0
            for i in range(len(label)):
                # 标注伪标签
                if (event[i] == 1):
                    pre_label.append(label[i].cpu().numpy())
                    pre_label_image.append(label[i].cpu().numpy())    
                    pre_label_text.append(label[i].cpu().numpy())                
                    # pre_label1.append(label_pred[i])
                    pre_label1 = torch.cat((pre_label1.to(device), label_pred[i].to(device)), 0)
                    pre_label1_image = torch.cat((pre_label1_image.to(device), pre_image[i].to(device)), 0)
                    pre_label1_text = torch.cat((pre_label1_text.to(device), pre_text[i].to(device)), 0)
                    num = num + 1

                else:
                    # 未标注数据，不计算交叉熵损失
                    if (event[i] == 3):
                        labeled3.append(label[i].cpu().numpy())
                        labeled_3 = torch.cat((labeled_3.to(device), label_pred[i].to(device)), 0)
                        num2 = num2 + 1
                    else:
                        # 事实标签，计算交叉熵损失
                        labeled.append(label[i].cpu().numpy())
                        labeled1 = torch.cat((labeled1.to(device), label_pred[i].to(device)), 0)

            # 转换为logit标签
            # 事实标签logit
            labeled2 = torch.reshape(labeled1, (int(len(labeled1) / 2), 2))
            labeled2 = labeled2[1:]
            # 伪标签logit
            pre_label2 = torch.reshape(pre_label1, (int(len(pre_label1) / 2), 2))
            pre_label2_image = torch.reshape(pre_label1_image, (int(len(pre_label1_image) / 2), 2))
            pre_label2_text = torch.reshape(pre_label1_text, (int(len(pre_label1_text) / 2), 2))
            pre_label2 = pre_label2[1:]
            pre_label2_image = pre_label2_image[1:]
            pre_label2_text = pre_label2_text[1:]
            # 未标注数据logit
            labeled4 = torch.reshape(labeled_3, (int(len(labeled_3) / 2), 2))
            labeled4 = labeled4[1:]

            # 转换为0/1标签
            # 事实标签
            labeled = np.array(labeled)
            labeled = torch.from_numpy(labeled)
            # 伪标签
            pre_label = np.array(pre_label)
            pre_label = torch.from_numpy(pre_label)
            pre_label_image = np.array(pre_label_image)
            pre_label_image = torch.from_numpy(pre_label_image)
            pre_label_text = np.array(pre_label_text)
            pre_label_text = torch.from_numpy(pre_label_text)
            # labeled3 = np.array(labeled3)
            # labeled3 = torch.from_numpy(labeled3)

            # tri_loss = loss_fn(label_pred.to(device), label.long().to(device))
            # 事实标签分类损失
            tri_loss1 = loss_fn(labeled2.to(device), labeled.long().to(device))

            # 伪标签分类损失
            if (num == 0):
                tri_loss2 = 0
                tri_loss2_image = 0
                tri_loss2_text = 0
            else:
                tri_loss2 = loss_fn(pre_label2.to(device), pre_label.long().to(device))
                tri_loss2_image = loss_fn(pre_label2_image.to(device), pre_label_image.long().to(device))
                tri_loss2_text = loss_fn(pre_label2_text.to(device), pre_label_text.long().to(device))
            loss = tri_loss1 + vsd_loss + tri_loss2 + loss_mse + tri_loss2_image + tri_loss2_text
        else:
            loss = 0

        return loss,label_pred,F_A,code_I,code_T

class teacher_model(nn.Module):
    def __init__(self):
        super(teacher_model, self).__init__()
        self.fcModal = fcModal()
        self.attention = Attention_ast(code_len=settings.CODE_LEN)
        self.VIB_I = VIB_I()
        self.VIB_T = VIB_T()
        self.kl_div = torch.nn.KLDivLoss()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, img,txt,label,event,step):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # add noise
        noise_std = 0.05
        img2 = img.cpu().numpy()
        txt2 = txt.cpu().numpy()
        img2 = img2 + noise_std * np.random.randn(*img2.shape)
        txt2 = txt2 + noise_std * np.random.randn(*txt2.shape)
        img3 = torch.from_numpy(img2)
        txt3 = torch.from_numpy(txt2)
        img3 = img3.to(torch.float32)
        txt3 = txt3.to(torch.float32)

        img,code_I = self.VIB_I(img)
        txt, code_T = self.VIB_T(txt)

        att_A1 = self.attention(code_I)
        att_A2 = self.attention(code_T)
        att_A = torch.cat((att_A1, att_A2), 1)
        att_A = torch.softmax(att_A, dim=1)
        att_A1 = att_A[:, :1]
        att_A2 = att_A[:, 1:]
        F_A = torch.multiply(att_A1, code_I) + torch.multiply(att_A2, code_T)
        label_pred, event_pred = self.fcModal(F_A, settings.BATCH_SIZE)
        # label_pred = F.softmax(label_pred, dim=1)



        # 生成噪声标签
        # txt3 = self.fc1(txt3.to(device))
        img3, code_I3 = self.VIB_I(img3.to(device))
        txt3, code_T3 = self.VIB_T(txt3.to(device))

        att_A13 = self.attention(code_I3)
        att_A23 = self.attention(code_T3)
        att_A3 = torch.cat((att_A13, att_A23), 1)
        att_A3 = torch.softmax(att_A3, dim=1)
        att_A13 = att_A3[:, :1]
        att_A23 = att_A3[:, 1:]
        F_A3 = torch.multiply(att_A13, code_I3) + torch.multiply(att_A23, code_T3)
        label_pred3, event_pred3 = self.fcModal(F_A3, settings.BATCH_SIZE)

        F_label_pred = F.normalize(label_pred)
        F_label_pred3 = F.normalize(label_pred3)
        #
        loss_mse = F.mse_loss(F_label_pred,F_label_pred3)
        # loss_mse = self.kl_div(input=self.softmax(F_label_pred3.detach() / 1),
        #             target=self.softmax(F_label_pred / 1))







        if(step=='train'):
            vsd_loss = self.kl_div(input=self.softmax(img.detach() / 1),
                                   target=self.softmax(code_I / 1)) + \
                       self.kl_div(input=self.softmax(txt.detach() / 1),
                                   target=self.softmax(code_T / 1))

            loss_fn = nn.CrossEntropyLoss()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            labeled = []
            labeled1 = torch.zeros(2)
            pre_label = []
            pre_label1 = torch.zeros(2)
            labeled3 = []
            labeled_3 = torch.zeros(2)
            num = 0
            num2 = 0
            for i in range(len(label)):
                if(event[i]==1):
                    pre_label.append(label[i].cpu().numpy())
                    # pre_label1.append(label_pred[i])
                    pre_label1 = torch.cat((pre_label1.to(device), label_pred[i].to(device)), 0)
                    num = num +1

                else:
                    if(event[i]==3):
                        labeled3.append(label[i].cpu().numpy())
                        labeled_3 = torch.cat((labeled_3.to(device), label_pred[i].to(device)), 0)
                        num2 = num2 + 1
                    else:
                        labeled.append(label[i].cpu().numpy())
                        labeled1 = torch.cat((labeled1.to(device), label_pred[i].to(device)), 0)

            labeled2 = torch.reshape(labeled1,(int(len(labeled1)/2),2))
            labeled2 = labeled2[1:]
            pre_label2 = torch.reshape(pre_label1,(int(len(pre_label1)/2),2))
            pre_label2 = pre_label2[1:]
            labeled4 = torch.reshape(labeled_3,(int(len(labeled_3)/2),2))
            labeled4 = labeled4[1:]

            labeled = np.array(labeled)
            labeled = torch.from_numpy(labeled)
            pre_label = np.array(pre_label)
            pre_label = torch.from_numpy(pre_label)
            labeled3 = np.array(labeled3)
            labeled3 = torch.from_numpy(labeled3)


            # tri_loss = loss_fn(label_pred.to(device), label.long().to(device))
            #事实标签分类损失
            tri_loss1 = loss_fn(labeled2.to(device), labeled.long().to(device))

            #伪标签分类损失
            if(num==0):
                tri_loss2 = 0
            else:
                tri_loss2 = loss_fn(pre_label2.to(device), pre_label.long().to(device))
            loss = tri_loss1  + vsd_loss + tri_loss2 + 0.1 * loss_mse
        else:
            loss = 0

        return loss,label_pred,F_A,code_I,code_T

class teacher_model2(nn.Module):
    def __init__(self):
        super(teacher_model2, self).__init__()
        self.fcModal = fcModal()
        self.attention = Attention_ast(code_len=settings.CODE_LEN)
        self.VIB_I = VIB_I()
        self.VIB_T = VIB_T()
        self.kl_div = torch.nn.KLDivLoss()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, img,txt,label,event,step):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # # add noise
        noise_std = 0.1
        img2 = img.cpu().numpy()
        txt2 = txt.cpu().numpy()
        img2 = img2 + noise_std * np.random.randn(*img2.shape)
        txt2 = txt2 + noise_std * np.random.randn(*txt2.shape)
        img3 = torch.from_numpy(img2)
        txt3 = torch.from_numpy(txt2)
        img3 = img3.to(torch.float32)
        txt3 = txt3.to(torch.float32)

        img,code_I = self.VIB_I(img)
        txt, code_T = self.VIB_T(txt)


        att_A1 = self.attention(code_I)
        att_A2 = self.attention(code_T)
        att_A = torch.cat((att_A1, att_A2), 1)
        att_A = torch.softmax(att_A, dim=1)
        att_A1 = att_A[:, :1]
        att_A2 = att_A[:, 1:]
        F_A = torch.multiply(att_A1, code_I) + torch.multiply(att_A2, code_T)
        label_pred, event_pred = self.fcModal(F_A, settings.BATCH_SIZE)
        # label_pred = F.softmax(label_pred, dim=1)


        # 生成噪声标签
        # txt3 = self.fc1(txt3.to(device))
        img3, code_I3 = self.VIB_I(img3.to(device))
        txt3, code_T3 = self.VIB_T(txt3.to(device))
        # txt3 = self.fc(txt3.to(device))
        # code_I3 = img3.to(device)
        # code_T3 = txt3.to(device)

        att_A13 = self.attention(code_I3)
        att_A23 = self.attention(code_T3)
        att_A3 = torch.cat((att_A13, att_A23), 1)
        att_A3 = torch.softmax(att_A3, dim=1)
        att_A13 = att_A3[:, :1]
        att_A23 = att_A3[:, 1:]
        F_A3 = torch.multiply(att_A13, code_I3) + torch.multiply(att_A23, code_T3)
        label_pred3, event_pred3 = self.fcModal(F_A3, settings.BATCH_SIZE)

        F_label_pred = F.normalize(label_pred)
        F_label_pred3 = F.normalize(label_pred3)
        #
        loss_mse = F.mse_loss(F_label_pred,F_label_pred3)


        if(step=='train'):
            vsd_loss = self.kl_div(input=self.softmax(img.detach() / 1),
                                   target=self.softmax(code_I / 1)) + \
                       self.kl_div(input=self.softmax(txt.detach() / 1),
                                   target=self.softmax(code_T / 1))

            loss_fn = nn.CrossEntropyLoss()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            labeled = []
            labeled1 = torch.zeros(2)
            pre_label = []
            pre_label1 = torch.zeros(2)
            labeled3 = []
            labeled_3 = torch.zeros(2)
            num = 0
            num2 = 0
            for i in range(len(label)):
                if(event[i]==1):
                    pre_label.append(label[i].cpu().numpy())
                    # pre_label1.append(label_pred[i])
                    pre_label1 = torch.cat((pre_label1.to(device), label_pred[i].to(device)), 0)
                    num = num +1

                else:
                    if(event[i]==3):
                        labeled3.append(label[i].cpu().numpy())
                        labeled_3 = torch.cat((labeled_3.to(device), label_pred[i].to(device)), 0)
                        num2 = num2 + 1
                    else:
                        labeled.append(label[i].cpu().numpy())
                        labeled1 = torch.cat((labeled1.to(device), label_pred[i].to(device)), 0)

            labeled2 = torch.reshape(labeled1,(int(len(labeled1)/2),2))
            labeled2 = labeled2[1:]
            pre_label2 = torch.reshape(pre_label1,(int(len(pre_label1)/2),2))
            pre_label2 = pre_label2[1:]
            labeled4 = torch.reshape(labeled_3,(int(len(labeled_3)/2),2))
            labeled4 = labeled4[1:]

            labeled = np.array(labeled)
            labeled = torch.from_numpy(labeled)
            pre_label = np.array(pre_label)
            pre_label = torch.from_numpy(pre_label)
            labeled3 = np.array(labeled3)
            labeled3 = torch.from_numpy(labeled3)


            # tri_loss = loss_fn(label_pred.to(device), label.long().to(device))
            #事实标签分类损失
            tri_loss1 = loss_fn(labeled2.to(device), labeled.long().to(device))

            #伪标签分类损失
            if(num==0):
                tri_loss2 = 0
            else:
                tri_loss2 = loss_fn(pre_label2.to(device), pre_label.long().to(device))
            loss = tri_loss1 + tri_loss2 + 0.1*loss_mse + vsd_loss
        else:
            loss = 0

        return loss,label_pred,F_A,code_I,code_T