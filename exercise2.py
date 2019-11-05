import keras
from keras.layers import LSTM, Input, Dense, Dropout, Embedding
from keras.models import Model
from keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np

# Read volume of electricity exchange
kpx = pd.read_excel(r'C:\Users\user\Desktop\kpx.xlsx', sheet_name='1', header = None)

# Select volume from excel file
kpx2017 = kpx.loc[7:1494, 2]
kpx2018 = kpx.loc[8047:10254, 2]
kpx2019 = kpx.loc[16807:, 2]

# Select sum of electricity exchange
sumkpx2017 = kpx.loc[7:1494, 3]
sumkpx2018 = kpx.loc[8047:10254, 3]
sumkpx2019 = kpx.loc[16807:, 3]

# Read wind speed observation values
ws2017 = pd.read_excel(r'C:\Users\user\Desktop\2017a.xlsx', sheet_name='1', header = None)
ws2018 = pd.read_excel(r'C:\Users\user\Desktop\2018a.xlsx', sheet_name='1', header = None)
ws2019 = pd.read_excel(r'C:\Users\user\Desktop\2019a.xlsx', sheet_name='1', header = None)

# Read wind direction observation values
wd2017 = pd.read_excel(r'C:\Users\user\Desktop\2017b.xlsx', sheet_name='1', header = None)
wd2018 = pd.read_excel(r'C:\Users\user\Desktop\2018b.xlsx', sheet_name='1', header = None)
wd2019 = pd.read_excel(r'C:\Users\user\Desktop\2019b.xlsx', sheet_name='1', header = None)

# Read date data
dd2017 = pd.read_excel(r'C:\Users\user\Desktop\2017b.xlsx', sheet_name='1', header = None)
dd2018 = pd.read_excel(r'C:\Users\user\Desktop\2018b.xlsx', sheet_name='1', header = None)
dd2019 = pd.read_excel(r'C:\Users\user\Desktop\2019b.xlsx', sheet_name='1', header = None)

# Select wind speed from excel files
ws2017 = ws2017.loc[721:,2]
ws2018 = ws2018.loc[1:,2]
ws2019 = ws2019.loc[1:,2]

# Select wind direction from excel files
wd2017 = wd2017.loc[721:,8]
wd2018 = wd2018.loc[1:,8]
wd2019 = wd2019.loc[1:,8]

# Select date data from excel files
dd2017 = dd2017.loc[721:,10]
dd2018 = dd2018.loc[1:,10]
dd2019 = dd2019.loc[1:,10]

# Make data set
kpxdata = np.concatenate((kpx2017, kpx2018, kpx2019))
sumkpxdata1 = np.concatenate((sumkpx2017, sumkpx2018, sumkpx2019))
wsdata = np.concatenate((ws2017, ws2018, ws2019))
wddata = np.concatenate((wd2017, wd2018, wd2019))
dddata = np.concatenate((dd2017, dd2018, dd2019))


#Select sum data
sumkpxdata = np.zeros(184)
for i in range(1,185,1):
	sumkpxdata[int(i-1)] = sumkpxdata1[int(24*i-24)]
sumkpxdata1 = None

# Reshape data to use in Min_Max Scaler
kpxdata = kpxdata.reshape(-1,1)
sumkpxdata = sumkpxdata.reshape(-1,1)
wsdata = wsdata.reshape(-1,1)
wddata = wddata.reshape(-1,1)
dddata = dddata.reshape(-1,1)

# Min_Max Scaler
from sklearn.preprocessing import MinMaxScaler

kpxscaler = MinMaxScaler()
sumkpxscaler = MinMaxScaler()
wsscaler = MinMaxScaler()
wdscaler = MinMaxScaler()
ddscaler = MinMaxScaler()

kpxscaler.fit(kpxdata)
sumkpxscaler.fit(sumkpxdata)
wsscaler.fit(wsdata)
wdscaler.fit(wddata)
ddscaler.fit(dddata)

kpxdata = kpxscaler.fit_transform(kpxdata)
sumkpxdata = sumkpxscaler.fit_transform(sumkpxdata)
wsdata = wsscaler.fit_transform(wsdata)
wddata = wdscaler.fit_transform(wddata)
dddata = ddscaler.fit_transform(dddata)

# Make input data to predict volume of exchange
wsinput = wsdata[4416:,:]
wdinput = wddata[4416:,:]
ddinput = dddata[4416:,:]
wsdata = wsdata[:4416,:]
wddata = wddata[:4416,:]
dddata = dddata[:4416,:]

# Make input data to put on machine
winput = np.hstack((wsinput, wdinput, ddinput))
wdata = np.hstack((wsdata, wddata, dddata))

# Reshape data set according to model input & output
seq_length = 24
input_dimension = 3
kpxdata = kpxdata.reshape(-1, seq_length)
sumkpxdata = sumkpxdata.reshape(-1, 1)
wdata = wdata.reshape(-1, seq_length, input_dimension)

# Shuffle batch
import random
rd = np.arange(0, 184, 1)
random.shuffle(rd)
kpxdata = kpxdata[rd,:]
sumkpxdata = sumkpxdata[rd,:]
wdata = wdata[rd,:,:]

# Make Training set & Validation set & Test set
train_size = 146
val_size = 19
test_size = 19

wTrain = wdata[:train_size, :, :]
wVal = wdata[train_size:int(train_size+val_size), :, :]
wTest = wdata[int(train_size+val_size):, :, :]
kpxTrain = kpxdata[:train_size, :]
kpxVal = kpxdata[train_size:int(train_size+val_size), :]
kpxTest = kpxdata[int(train_size+val_size):, :]
sumkpxTrain = sumkpxdata[:train_size, :]
sumkpxVal = sumkpxdata[train_size:int(train_size+val_size), :]
sumkpxTest = sumkpxdata[int(train_size+val_size):, :]

# Make shared-LSTM layer
sharedLSTM1 = LSTM(72, input_shape=(seq_length, input_dimension), return_sequences=True)
sharedLSTM2 = LSTM(64)
# Model 1
inputLayer1 = Input(shape=(seq_length, input_dimension))
sharedLSTM1Instance1 = sharedLSTM1(inputLayer1)
sharedLSTM2Instance1 = sharedLSTM2(sharedLSTM1Instance1)
dropoutLayer1 = Dropout(0.1)(sharedLSTM2Instance1)
denseLayer11 = Dense(64)(dropoutLayer1)
denseLayer21 = Dense(64)(denseLayer11)
outputLayer1 = Dense(1)(denseLayer21)

model1 = Model(inputs=inputLayer1, outputs=outputLayer1)
model1.compile(loss='mean_squared_error', optimizer='adagrad')

# Model 2
inputLayer = Input(shape=(seq_length, input_dimension))
sharedLSTM1Instance = sharedLSTM1(inputLayer)
sharedLSTM2Instance = sharedLSTM2(sharedLSTM1Instance)
dropoutLayer = Dropout(0.1)(sharedLSTM2Instance)
denseLayer1 = Dense(128)(dropoutLayer)
denseLayer2 = Dense(128)(denseLayer1)
outputLayer = Dense(24)(denseLayer2)

model = Model(inputs=inputLayer, outputs=outputLayer)
model.compile(loss='mean_squared_error', optimizer='adagrad')

# Training
early_stopping = EarlyStopping(patience = 5)
hist1 = model1.fit(wTrain, sumkpxTrain, epochs=100, batch_size = 2, validation_data = (wVal, sumkpxVal), callbacks=[early_stopping])
hist = model.fit(wTrain, kpxTrain, epochs=100, batch_size = 2, validation_data = (wVal, kpxVal), callbacks=[early_stopping])

# Training Score
trainScore1 = model1.evaluate(wTrain, sumkpxTrain)
print(trainScore1)
model1.reset_states()
testScore1 = model1.evaluate(wTest, sumkpxTest)
print(testScore1)
model1.reset_states()

trainScore = model.evaluate(wTrain, kpxTrain)
print(trainScore)
model.reset_states()
testScore = model.evaluate(wTest, kpxTest)
print(testScore)
model.reset_states()

# Make an excel file & Write result data
winput = winput.reshape(-1, seq_length, input_dimension)
yhat = model.predict(winput, verbose=0)
yhat[yhat < 0] = 0

yhat = yhat.reshape(-1,1)

yhat = kpxscaler.inverse_transform(yhat)

import xlrd, xlwt
workbookw = xlwt.Workbook(encoding='utf-8')
workbookw.default_style.font.height = 20 * 11
worksheetw = workbookw.add_sheet('1')
for row_num in range(744):
    worksheetw.write(row_num, 0, float(yhat[row_num][0]))

workbookw.save(r'C:\Users\user\Desktop\result.xls')
