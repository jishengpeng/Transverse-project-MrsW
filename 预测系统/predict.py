import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.autograd import Variable
import csv
import argparse
import logging
from sklearn.model_selection import train_test_split

#超参数和全局变量
parser = argparse.ArgumentParser("学习满意度")
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--epochs', type=int, default=5, help='num of training epochs')
parser.add_argument('--kernel_number', type=int, default=16, help='number of kernel')
parser.add_argument('--hidden_size', type=int, default=128, help='number of hidden cell')
args = parser.parse_args()


#日志
logging.basicConfig(level=logging.INFO,
filename="D:\\bianyiqi\\学习满意度数据挖掘\\预测系统\\训练神经网络日志.txt",
filemode='a',
format="%(asctime)s %(message)s")


#定义输出文件
header=['预测值','真实值']
log_path = '预测结果.csv'
file = open(log_path, 'w',  newline='',encoding="utf-8")
csv_writer = csv.writer(file)
csv_writer.writerow(header)

#定义设备
if(torch.cuda.is_available()):
    device=torch.device("cuda:0")
else:
    device=torch.device("cpu")

class Dataset(torch.utils.data.Dataset):
	def __init__(self, x, label):
		super(Dataset, self).__init__()
		self.x = x
		self.label = label

	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		return self.x[idx], self.label[idx]

#搭建网络
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.embedding=nn.Embedding(10,4)   #前一个指编码中的最大数，后一个指第三维是多少
        #卷积层，第一个是卷积核，然后最大池化，然后
        self.Cnn_model5=nn.Sequential(
            nn.Conv1d(in_channels=53,out_channels=args.kernel_number,kernel_size=5,stride=1,padding=2),#我这里还是第二维是域名的长度，第三维是特征，out_channel理解为卷积核的个,卷积是在第二维度
            nn.MaxPool1d(kernel_size=2),  #这个地方就是第三个维度直接用输入除以输出
            nn.BatchNorm1d(args.kernel_number)  #进行标准化

        )
        self.Cnn_model3=nn.Sequential(
            nn.Conv1d(in_channels=53,out_channels=args.kernel_number,kernel_size=3,stride=1),#我这里还是第二维是域名的长度，第三维是特征，out_channel理解为卷积核的个
            nn.MaxPool1d(kernel_size=2) , #这个地方就是第三个维度直接用输入除以输出
            nn.BatchNorm1d(args.kernel_number)
        )
        self.Cnn_model4=nn.Sequential(
            nn.Conv1d(in_channels=53,out_channels=args.kernel_number,kernel_size=4,stride=1),#我这里还是第二维是域名的长度，第三维是特征，out_channel理解为卷积核的个
            nn.MaxPool1d(kernel_size=2) , #这个地方就是第三个维度直接用输入除以输出
            nn.BatchNorm1d(args.kernel_number)
        )


        #再来一些循环神经网络
        self.lstm1 = nn.LSTM(input_size=381, hidden_size=args.hidden_size, num_layers=1, batch_first=True,
                             bidirectional=False)  # batch_first表示是否将batch放在第一个,hidden_size表示多少个神经元

        # 再来一些循环神经网络
        self.lstm2 = nn.LSTM(input_size=args.hidden_size, hidden_size=32, num_layers=1, batch_first=True,
                            bidirectional=False)

        #防止过拟合
        self.dropout = nn.Dropout(p=0.2)

        #
        self.flatten=nn.Flatten()

        self.linear_model = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Linear(32, 13),
            nn.Softmax()
            # nn.Sigmoid(),
            # nn.Linear(16, 4),
            # nn.Sigmoid(),
            # nn.Linear(4, 1)
        )
        self.linear_model2 = nn.Sequential(
            nn.Linear(53, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Linear(32, 13),
            nn.Softmax()
            # nn.Sigmoid(),
            # nn.Linear(16, 4),
            # nn.Sigmoid(),
            # nn.Linear(4, 1)
        )
        # self.softmax=nn.Softmax(dim=1)



    def forward(self,x):
        # x=self.embedding(x)      #维度是(batch_size,seq_length,embedding_dim)

        # x3=self.Cnn_model3(x)
        # x4=self.Cnn_model4(x)
        # x5=self.Cnn_model5(x)
        # #
        # x0=torch.cat((x5,x3,x4),dim=2)   #这个地方是最终的输入维度（batch_size,max_document_length,310）
        #
        # output,(h_n,c_n)=self.lstm1(x0)
        # output, (h_n, c_n) = self.lstm2(output)
        # output=self.dropout(output)
        #
        # #最后的解码器部分
        # output=self.flatten(output)    #维度是(batch_size,max_document_length*hidden_size)
        # output=self.linear_model(output)
        # output = self.flatten(x)
        output = self.linear_model2(x)
        return output


def get_data():
    df=pd.read_csv("D:\\bianyiqi\\学习满意度数据挖掘\\预测系统\\预测数据.csv")
    data=np.array(df.values)
    height,weight=data.shape
    len=height
    #输入和输出
    y=[]
    x=[]
    for i in range(0,len):
        y.append(data[i][0]-3)
        tmp=[]
        for j in range(1,54):
            tmp.append(data[i][j])
        x.append(tmp)

    # 这个地方我提前将其变成tensor张量，后面变非常麻烦
    y = torch.Tensor(y)
    x = torch.Tensor(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=20)  # 划分训练集和测试集，并且设置随机种子
    # print(x_train)
    return x_train, x_test, y_train, y_test


def main():
    get_data()
    train_x, test_x, train_label, test_label = get_data()

    # 数据加载
    train_data = Dataset(x=train_x, label=train_label)
    valid_data = Dataset(x=test_x, label=test_label)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=True)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=True)

    # 定义模型
    model = Model()
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    # #这个地方为了搭建网络模型结构进行局部测试
    # for data in train_queue:
    #     input,label=data
    #     input = Variable(input)
    #     label = Variable(label)
    #     input=input.to(device)
    #     label=label.to(device)
    #     output=model(input)
    #     print(output.shape)

    # 开始训练
    for i in range(args.epochs):
        # 记录训练的次数
        total_trainstep = 0
        # 记录测试的次数
        total_validstep = 0
        print("--------------第{}轮训练开始--------------".format(i + 1))
        logging.info("--------------第{}轮训练开始--------------".format(i + 1))

        model.train()
        for data in train_queue:
            input, label = data
            input = Variable(input)
            label = Variable(label)
            input = input.to(device)
            label = label.to(device)
            output = model(input)
            # print(output.shape)

            # print(label)
            # pre=output.argmax(1)
            # print(pre)
            loss = loss_fn(output, label.long())

            # print(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_trainstep = total_trainstep + 1

            if (total_trainstep % 1 == 0):
                print("训练次数：{} , loss :{}".format(total_trainstep, loss))
                logging.info("训练次数：{} , loss :{}".format(total_trainstep, loss))

        # 测试步骤
        model.eval()
        with torch.no_grad():
            for data1 in valid_queue:
                final_true = []
                final_pridict = []
                input1, label1 = data1
                input1 = Variable(input1)
                label1 = Variable(label1)
                input1 = input1.to(device)
                label1 = label1.to(device)

                output1 = model(input1)
                loss1 = loss_fn(output1, label1.long())
                print(output1)
                # print(label1)
                # for ii in range(0, args.batch_size):
                #     final_pridict.append(output1[ii][0])
                #     final_true.append(label1[ii])
                for ii in range(0, args.batch_size):
                    final_pridict.append(output1.cpu().argmax(1)[ii])
                    final_true.append(label1.cpu()[ii])
            print(final_pridict,final_true)
        #             #写进文件中
        #             if(i==args.epochs-1):
        #                 final_write = []
        #                 final_write.append(output1[ii][0])
        #                 final_write.append(label1[ii])
        #                 csv_writer.writerow(final_write)
        #
        # #
        #         total_validstep = total_validstep + 1
        #
        #         if (total_validstep % 5 == 0):
        #             print("测试次数：{} , 预测情况 :{} , 实际情况:{}".format(total_validstep, final_pridict,final_true))
        #             logging.info("测试次数：{} , 预测情况 :{}, 实际情况:{}".format(total_validstep, final_pridict,final_true))




if __name__=="__main__":
    main()
    file.close()
