import librosa
import torch
import torch.nn as nn
import numpy as np
import os
import random
from torch.utils.data import TensorDataset, DataLoader

Machine_path = r"C:\Users\msi-pc\Desktop\Datas\sorted\Machine"
Men_path = r"C:\Users\msi-pc\Desktop\Datas\sorted\People"

Machine_file = os.listdir(Machine_path)
Men_file = os.listdir(Men_path)
Machine_list = []
Men_list = []

for i in Machine_file:
    Machine_filename = Machine_path + '\\' + i
    wav_file,sr = librosa.load(Machine_filename)
    spec_mag = librosa.feature.melspectrogram(y=wav_file,sr=sr,hop_length=2048)
    spec_mag = spec_mag.flatten()
    if len(spec_mag) > 2000:
        spec_mag = spec_mag[:2000]
    elif len(spec_mag) < 2000:
        spec_mag = list(spec_mag)
        while len(spec_mag) < 2000:
            spec_mag.append(0)
    Machine_list.append(spec_mag)


for i in Men_file:
    Men_filename = Men_path + '\\' + i
    wav_file, sr = librosa.load(Men_filename)
    spec_mag = librosa.feature.melspectrogram(y=wav_file,hop_length=2048)
    spec_mag = spec_mag.flatten()
    spec_mag = spec_mag[2000:]
    if len(spec_mag) > 2000:
        spec_mag = spec_mag[:2000]
    elif len(spec_mag) < 2000:
        spec_mag = list(spec_mag)
        while len(spec_mag) < 2000:
            spec_mag.append(0)
    Men_list.append(spec_mag)

Men_label = np.zeros(len(Men_list))#人0机1
Machine_lable = np.ones(len(Machine_list))

X_train_data = []
Y_train_data = []
X_train_data.append(Men_list)
Y_train_data.append(Men_label)
X_train_data.append(Machine_list)
Y_train_data.append(Machine_lable)
X_train_data = np.array(X_train_data,dtype=np.float32)
Y_train_data = np.array(Y_train_data,dtype=np.float32)
print(X_train_data.shape)
print(Y_train_data.shape)
X_train_data = X_train_data.reshape(2*X_train_data.shape[1],X_train_data.shape[2])
Y_train_data = Y_train_data.reshape(Y_train_data.shape[0]*Y_train_data.shape[1])
print(X_train_data.shape)
print(Y_train_data.shape)
input_size = len(X_train_data[0])
output_size = 1

#打乱顺序
Zip = list(zip(X_train_data, Y_train_data))
random.shuffle(Zip)
X_train_data, Y_train_data = zip(*Zip)
X_train_data = list(X_train_data)
Y_train_data = list(Y_train_data)
X_train_data = np.array(X_train_data,dtype=np.float32)
Y_train_data = np.array(Y_train_data,dtype=np.float32)
X_train_data = torch.tensor(X_train_data).cuda()
Y_train_data = torch.tensor(Y_train_data).cuda()
#划分验证集
ratio = 0.9
train_size = int(len(X_train_data)*ratio)
val_size = len(X_train_data) - train_size

X_Train = X_train_data[:train_size]
X_Val = X_train_data[train_size:]
Y_Train = Y_train_data[:train_size]
Y_Val = Y_train_data[train_size:]
print(X_Train.size())
print(X_Val.size())
print(Y_Train.size())
print(Y_Val.size())

#加载数据集
Train_dataset = TensorDataset(X_Train, Y_Train)
Val_dataset = TensorDataset(X_Val, Y_Val)
batch_size = 32

Train_loader = DataLoader(Train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
Val_loader = DataLoader(Val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

#Step3. 构建模型
class Classification(nn.Module):
    def __init__(self, input_size, output_size):
        super(Classification, self).__init__()
        self.input_size = input_size
        self.out_size = output_size
        self.linear1 = nn.Linear(input_size,128)#Question:中间层神经元的数量怎么确定？
        self.linear2 = nn.Linear(128, output_size)
        self.dropout = nn.Dropout(p=0.5) 
        self.sigmoid = nn.Sigmoid()
        self.ReLU = nn.ReLU()
    def forward(self, x):
        result1 = self.linear1(x)
        y_pred1 = self.sigmoid(result1)
        result2 = self.linear2(y_pred1)
        y_pred2 = self.sigmoid(result2)
        #y_pred = self.ReLU(self.linear2(self.ReLU(self.linear1(x))))
        y_pred2 = y_pred2.squeeze(-1)#为什么标准化就不需要这一句？？？？
        return y_pred2

#实例化对象
Classification_Model = Classification(input_size, output_size)
Classification_Model.cuda()

#初始化超参数
creterion = nn.BCELoss()
optimizer = torch.optim.Adam(Classification_Model.parameters(), lr = 0.00003)
epoches = 100

#Step4. 训练函数
def train(model, train_loader, val_loader, creterion, epoches, optimizer):
    for epoch in range(epoches):
        model.train()
        train_loss = []
        num_correct = 0
        for x, y_real in train_loader:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = creterion(y_pred, y_real)
            train_loss.append(loss.item())#item的作用是把tensor\ndarray里的值取出来(当tensor\ndarray中只有一个元素时)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = []
        for x, y_real in val_loader:
            #print("len of y-real is %d"%len(y_real))
            y_pred = model(x)
            y_pred = model(x)
            y_pred_sigmoid = torch.round(y_pred)
            loss = creterion(y_pred, y_real)
            correct = torch.eq(y_pred_sigmoid,y_real)
            correct = correct.cpu().numpy()
            result = np.sum(correct)
            num_correct += result
            val_loss.append(loss.item())
        print(f'Epoch {epoch+1}/{epoches} --- train loss {np.round(np.mean(train_loss), 5)} --- val loss {np.round(np.mean(val_loss), 5)}')
        print('Total test accuracy is {:.2f}'.format(np.mean(num_correct/len(val_loader.dataset))))

train(Classification_Model, Train_loader, Val_loader, creterion, epoches, optimizer)
