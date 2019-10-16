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
kpx2018 = kpx.loc[8047:10254,2]
kpx2019 = kpx.loc[16807:,2]

# Read wind speed observation values
ws2017 = pd.read_excel(r'C:\Users\user\Desktop\2017a.xlsx', sheet_name='1', header = None)
ws2018 = pd.read_excel(r'C:\Users\user\Desktop\2018a.xlsx', sheet_name='1', header = None)
ws2019 = pd.read_excel(r'C:\Users\user\Desktop\2019a.xlsx', sheet_name='1', header = None)

# Read wind direction observation values
wd2017 = pd.read_excel(r'C:\Users\user\Desktop\2017b.xlsx', sheet_name='1', header = None)
wd2018 = pd.read_excel(r'C:\Users\user\Desktop\2018b.xlsx', sheet_name='1', header = None)
wd2019 = pd.read_excel(r'C:\Users\user\Desktop\2019b.xlsx', sheet_name='1', header = None)

# Select wind speed from excel files
ws2017 = ws2017.loc[721:,2]
ws2018 = ws2018.loc[1:,2]
ws2019 = ws2019.loc[1:,2]

# Select wind direction from excel files
wd2017 = wd2017.loc[721:,2]
wd2018 = wd2018.loc[1:,2]
wd2019 = wd2019.loc[1:,2]

# Make data set
kpxdata = np.concatenate((kpx2017, kpx2018, kpx2019))
wsdata = np.concatenate((ws2017, ws2018, ws2019))
wddata = np.concatenate((wd2017, wd2018, wd2019))

# Min_Max Scaler
from sklearn.preprocessing import MinMaxScaler

kpxdata = kpxdata.reshape(-1,1)
wsdata = wsdata.reshape(-1,1)
wddata = wddata.reshape(-1,1)
kpxscaler = MinMaxScaler()
wsscaler = MinMaxScaler()
wdscaler = MinMaxScaler()
kpxscaler.fit(kpxdata)
wsscaler.fit(wsdata)
wdscaler.fit(wddata)
kpxdata = kpxscaler.fit_transform(kpxdata)
wsdata = wsscaler.fit_transform(wsdata)
wddata = wdscaler.fit_transform(wddata)

# Make input data to predict volume of exchange
wsinput = wsdata[4416:,:]
wdinput = wddata[4416:,:]
wsdata = wsdata[:4416,:]
wddata = wddata[:4416,:]

# Make input data to put on machine
wdata = np.hstack([wsdata, wddata])
winput = np.hstack([wsinput, wdinput])

# Reshape data set according to model input & output
seq_length = 24
input_dimension = 2
kpxdata = kpxdata.reshape(-1, 24)
wdata = wdata.reshape(-1, seq_length, input_dimension)

# Shuffle batch
import random
rd = np.arange(0, 184, 1)
random.shuffle(rd)
kpxdata = kpxdata[rd,:]
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

# Model
sharedLSTM1 = LSTM(24, input_shape=(24, 2), return_sequences=True)
sharedLSTM2 = LSTM(48)

inputLayer = Input(shape=(24, 2))
sharedLSTM1Instance = sharedLSTM1(inputLayer)
sharedLSTM2Instance = sharedLSTM2(sharedLSTM1Instance)
dropoutLayer = Dropout(0.1)(sharedLSTM2Instance)
denseLayer1 = Dense(16)(dropoutLayer)
denseLayer2 = Dense(32)(denseLayer1)
outputLayer = Dense(24)(denseLayer2)

model = Model(inputs=inputLayer, outputs=outputLayer)
model.compile(loss='mean_squared_error', optimizer='adagrad')

# Training
early_stopping = EarlyStopping(patience = 5)
hist = model.fit(wTrain, kpxTrain, epochs=100, batch_size = 4, validation_data = (wVal, kpxVal), callbacks=[early_stopping])

trainScore = model.evaluate(wTrain, kpxTrain)
print(trainScore)
model.reset_states()
testScore = model.evaluate(wTest, kpxTest)
print(testScore)
model.reset_states()

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