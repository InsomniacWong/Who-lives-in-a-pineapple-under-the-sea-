import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
from string import punctuation
import jieba

f_path = r"C:\Users\msi-pc\Desktop\小创\垃圾短信合集.txt"
stop_path = r"C:\Users\msi-pc\Desktop\小创\中文停用词.txt"
pretrained_path = r"C:\Users\msi-pc\Desktop\zhwiki\vectors.txt"
embeds_size = 50
output_size = 1

def Load_file(f_path):
    label_list = []
    content_list = []
    with open(f_path) as f:
        for line in f:
            splited_line = line.split()
            label_list.append(splited_line[0])
            content_list.append(splited_line[1])
    label_list = np.array(label_list)
    content_list = np.array(content_list)
    return content_list, label_list

def Get_Content(content_list, label_list):
    Junk = []
    Good = []
    for i in range(len(content_list)):
        if(label_list[i] == '1'):
            Junk.append(content_list[i])
        if(label_list[i] == '0'):
            Good.append(content_list[i])
    Junk = np.array(Junk)
    Good = np.array(Good)
    return Junk[:50000], Good[:50000]

def read_stopword(stop_path):
    with open(stop_path, 'r', encoding='utf-8') as file:
        stopword = file.readlines()
    return [word.replace('\n', '') for word in stopword]

def Get_words(Content, stop_word):
    content = ','.join(list(Content))
    wordlist = jieba.lcut(content)#切割词语
    new_wordlist = []
    for word in wordlist:
        if word not in stop_word and word not in punctuation and len(word) > 1 and 'x' not in word:
            new_wordlist.append(word)
    return new_wordlist

def Generate_data(Junk_word, Good_word):#0代表好词，1代表垃圾词
    Good_list = Counter(Good_word).most_common(3000)
    Junk_list = Counter(Junk_word).most_common(3000)
    Good_dict = {}
    Junk_dict = {}
    word_list = []
    label_list = []
    for i in Good_list:
         Good_dict[i[0]] = i[1]
    for i in Junk_list:
         Junk_dict[i[0]] = i[1]
    Good_keys = Good_dict.keys()
    Junk_keys = Junk_dict.keys()
    for key in Good_keys:
        word_list.append(key)
        if key in Junk_keys:
            if(Good_dict[key] > Junk_dict[key]):
                label_list.append(0)
            else:
                label_list.append(1)
        else:
            label_list.append(0)
    for key in Junk_keys:
        if key not in word_list:
            word_list.append(key)
            label_list.append(1)
    return word_list, label_list

def Word2int(word_list):
    word_dict = dict(enumerate(word_list,1))
    word_dict = {w:int(i) for i, w in word_dict.items()}#颠倒这个字典的键和值，也是为了word2int省事儿
    return word_dict.values()
    
def Load_Glove(word_list, Glove_path):#注意这里的输入是word_list
    glove_list = []
    new_word_list = []
    with open(Glove_path) as f:
        for line in f:
            line_sp = line.split()
            if line_sp[0] in word_list:
                glove_list.append(line_sp)
    vec_list = []
    for i in word_list:
        for j in glove_list:
            if(i == j[0]):
                new_word_list.append(i)
                vec_list.append(j[1:])
    vec_list.append(glove_list[0][1:])
    vec_list = np.array(vec_list).astype(float)
    vec_list = torch.tensor(vec_list)
    return vec_list, new_word_list

def shuffle(word_list, label_list):
    #打乱顺序
    Zip = list(zip(word_list, label_list))
    random.shuffle(Zip)
    word_list, label_list = zip(*Zip)
    return word_list, label_list

def divide(word_list, label_list, ratio):
    word_list = list(word_list)
    train_size = int(len(word_list)*ratio)
    train_word = word_list[:train_size]
    train_label = label_list[:train_size]
    val_word = word_list[train_size:]
    val_label = label_list[train_size:]
    return train_word, train_label, val_word, val_label

def Load_data(train_word, train_label, val_word, val_label):
    train_word = np.array(train_word)
    train_label = np.array(train_label)
    val_word = np.array(val_word)
    val_label = np.array(val_label)
    train_word = torch.tensor(train_word)
    train_label = torch.tensor(train_label)
    val_word = torch.tensor(val_word)
    val_label = torch.tensor(val_label)
    train_dataset = TensorDataset(train_word,train_label)#把标签和数据集组合封装
    val_dataset = TensorDataset(val_word,val_label)
    train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True,pin_memory=True,drop_last=True)
    val_loader = DataLoader(val_dataset,batch_size=64,shuffle=True,pin_memory=True,drop_last=True)
    return train_loader, val_loader

class Classification(nn.Module):
    def __init__(self, input_size, output_size, glove):
        super(Classification, self).__init__()
        self.input_size = input_size
        self.out_size = output_size
        self.embedding = nn.Embedding.from_pretrained(glove, freeze=False)
        self.linear1 = nn.Linear(input_size,128)#Question:中间层神经元的数量怎么确定？
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 128)
        self.linear4 = nn.Linear(128, 128)
        self.linear5 = nn.Linear(128, output_size)
        self.dropout = nn.Dropout(p=0.5) 
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.embedding(x)
        x = x.to(torch.float32)
        result1 = self.linear1(x)
        y_pred1 = self.sigmoid(result1)
        result2 = self.linear2(y_pred1)
        y_pred2 = self.sigmoid(result2)
        result3 = self.linear3(y_pred2)
        y_pred3 = self.sigmoid(result3)
        result4 = self.linear4(y_pred3)
        y_pred4 = self.sigmoid(result4)
        result5 = self.linear5(y_pred4)
        y_pred5 = self.sigmoid(result5)
        y_pred5 = y_pred5.squeeze(-1)
        return y_pred5

def train(model, train_loader, val_loader, creterion, epoches, optimizer):
    for epoch in range(epoches):
        model.train()
        train_loss = []
        num_correct = 0
        for x, y_real in train_loader:
            optimizer.zero_grad()
            y_pred = model(x)
            y_real = y_real.to(torch.float32)
            loss = creterion(y_pred, y_real)
            train_loss.append(loss.item())#item的作用是把tensor\ndarray里的值取出来(当tensor\ndarray中只有一个元素时)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = []
        for x, y_real in val_loader:
            #print("len of y-real is %d"%len(y_real))
            y_pred = model(x)
            y_real = y_real.to(torch.float32)
            y_pred_sigmoid = torch.round(y_pred)
            loss = creterion(y_pred, y_real)
            correct = torch.eq(y_pred_sigmoid,y_real)
            correct = correct.cpu().numpy()
            result = np.sum(correct)
            num_correct += result
            val_loss.append(loss.item())
        print(f'Epoch {epoch+1}/{epoches} --- train loss {np.round(np.mean(train_loss), 5)} --- val loss {np.round(np.mean(val_loss), 5)}')
        print('Total test accuracy is {:.2f}'.format(np.mean(num_correct/len(val_loader.dataset))))

a, b = Load_file(f_path)
c, d = Get_Content(a,b)
Good_words = Get_words(d, stop_path)
Junk_words = Get_words(c, stop_path)
e, f = Generate_data(Good_words, Junk_words)
glove, e = Load_Glove(e, pretrained_path)
e = Word2int(e)
print(len(e))
g, h = shuffle(e, f)
i, j, k, l = divide(g, h, 0.9)
m, n = Load_data(i, j, k, l)

#实例化对象
Classification_Model = Classification(embeds_size, output_size, glove)

#初始化超参数
creterion = nn.BCELoss()
optimizer = torch.optim.Adam(Classification_Model.parameters(), lr = 0.00001)
epoches = 500

train(Classification_Model, m, n, creterion, epoches, optimizer)

