import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super().__init__()

        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


class LSTMnb:
    def __init__(self, data, l, r, n, train_window):
        self.data = data
        self.l = l
        self.r = r
        self.n = n
        self.train_window = train_window

    def train_and_predict(self, learning_rate=0.001, epochs=50):
        # 1. 获取训练集
        all_data = self.data['Value'].values.astype(float)
        train_data = all_data[self.l:self.r]

        # 2. 标准化并转为浮点数
        scaler = MinMaxScaler(feature_range=(-1, 1))
        train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1))
        train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)

        # 3. 根据train_window切分训练集获取喂入模型的数据
        train_inout_seq = []
        L = len(train_data_normalized)
        for i in range(L - self.train_window):
            train_seq = train_data_normalized[i: i + self.train_window]
            train_label = train_data_normalized[i + self.train_window:i + self.train_window + 1]
            train_inout_seq.append((train_seq, train_label))

        # 4. 初始化lstm
        model = LSTM()
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # 5. 开始训练
        for i in range(epochs):
            for seq, labels in train_inout_seq:
                optimizer.zero_grad()
                model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                     torch.zeros(1, 1, model.hidden_layer_size))

                y_pred = model(seq)

                single_loss = loss_function(y_pred, labels)
                single_loss.backward()
                optimizer.step()

            if i % 10 == 0:
                print('epoch:{:3} loss:{:10.8f}'.format(i, single_loss.item()))

        # 6. 进行预测
        test_inputs = train_data_normalized[-self.n:].tolist()

        model.eval()

        for i in range(self.n):
            seq = torch.FloatTensor(test_inputs)
            with torch.no_grad():
                model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                                torch.zeros(1, 1, model.hidden_layer_size))
                test_inputs.append(model(seq).item())

        # 7. 反标准化恢复数值
        self.pred = scaler.inverse_transform(np.array(test_inputs[-self.n:]).reshape(-1, 1))

        return self.pred

    def plot(self):
        x = range(self.r, self.r + self.n)
        plt.title('LSTM nb!')
        plt.ylabel('Value')
        plt.xlabel('time')
        plt.grid(True)
        plt.autoscale(axis='x', tight=True)
        plt.plot(data['Value'][self.l:self.r + self.n])
        plt.plot(x, self.pred)
        plt.show()


if __name__ == '__main__':
    data = pd.read_csv('BCHAIN-MKPRU.csv')
    # xx = len(data)  # MAX = 1826
    xx = 100  # 预测到第几个
    s = 14  # 几个为一组预测
    t = 1  # 预测几个

    T1 = time.time()

    res = []
    judge = []
    for i in range(xx - s):
        print("###############")
        print("第", i + s, "个数据开始预测")
        model = LSTMnb(data=data, l=i, r=i + s, n=t, train_window=7)
        pred = model.train_and_predict(learning_rate=0.001, epochs=50)

        res.append(pred[0][0])
        # 自制评价指标 增长率差值
        judge.append(abs(pred[0][0] - data['Value'][i + s].item()) / data['Value'][i + s - 1].item())

    print(judge)
    print(sum(judge) / len(judge))

    # 绘制数据
    x = range(s, xx)
    plt.plot(data['Value'][0:xx])
    plt.plot(x, res)
    plt.grid(True)
    plt.title("result")
    plt.ylabel("Value")
    plt.xlabel("Time")
    plt.autoscale(axis='x', tight=True)
    plt.show()

    # save = pd.DataFrame(res, columns=['Predict'])
    # save['Date'] = list(data['Date'][s:xx])
    # save['Value'] = list(data['Value'][s:xx])
    # save.to_csv('bchain.csv', index=None)

    T2 = time.time()
    print('程序运行时间{}秒'.format(T2 - T1))
