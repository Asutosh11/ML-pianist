#!/usr/bin/env python
# coding: utf-8

# In[1]:


# pip install tensorflow-gpu keras sklearn mido

import mido
from mido import MidiFile, MidiTrack, Message
from keras.layers import LSTM, Dense, Activation, Dropout, Flatten
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import numpy as np


# In[5]:


mid_train = MidiFile('piano_sonata_310_1_(c)oguri.mid') 
notes = []

notes = []
for msg in mid_train:
    if not msg.is_meta and msg.channel == 0 and msg.type == 'note_on':
        data = msg.bytes()
        notes.append(data[1])
        
scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(np.array(notes).reshape(-1,1))
notes = list(scaler.transform(np.array(notes).reshape(-1,1)))

notes = [list(note) for note in notes]

X = []
y = []
n_prev = 30
for i in range(len(notes)-n_prev):
    X.append(notes[i:i+n_prev])
    y.append(notes[i+n_prev])


# In[6]:


model = Sequential()

model.add(LSTM(256, input_shape=(n_prev, 1), return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(256, input_shape=(n_prev, 1), return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(256, input_shape=(n_prev, 1), return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(256, input_shape=(n_prev, 1), return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(256, input_shape=(n_prev, 1), return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(256, input_shape=(n_prev, 1), return_sequences=False))
model.add(Dropout(0.3))

model.add(Dense(1))

model.add(Activation('softmax'))
optimizer = Adam(lr=0.001)

model.compile(loss='mse', optimizer=optimizer)

model.fit(np.array(X), np.array(y), 16, 20, verbose=1, callbacks=[model_save_callback])


# In[25]:


mid_compose = MidiFile('mozart_25_1st_orch.mid') 
notes_compose = []
for msg in mid_compose:
    if not msg.is_meta and msg.channel == 0 and msg.type == 'note_on':
        data = msg.bytes()
        notes_compose.append(data[1])
        
scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(np.array(notes_compose).reshape(-1,1))
notes_compose = list(scaler.transform(np.array(notes_compose).reshape(-1,1)))

notes_compose = [list(note) for note in notes_compose]

X_compose = []
n_prev = 30

for i in range(len(notes_compose)-n_prev):
    X_compose.append(notes_compose[i:i+n_prev])
    
prediction = model.predict(np.array(X_compose))
prediction = np.squeeze(prediction)
prediction = np.squeeze(scaler.inverse_transform(prediction.reshape(-1,1)))
prediction = [int(i) for i in prediction]


# #### We have to save the generated sequence to a .mid file which can be converted to an mp3 easily

# In[26]:


mid_compose = MidiFile()
track = MidiTrack()
t = 0
for note in prediction:
    note = np.asarray([147, note, 67])
    bytes = note.astype(int)
    msg = Message.from_bytes(bytes[0:3])
    t += 1
    msg.time = t
    track.append(msg)
mid_compose.tracks.append(track)
mid_compose.save('generated_music.mid')

