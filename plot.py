import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('history/model_small_04_horizon40.csv')

df = df[1:]

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.plot(df['epoch'], df['loss'], label='loss') 
ax.plot(df['epoch'], df['val_loss'], label='val_loss') 

plt.title('Training and validation loss')
plt.xlabel('Epochs')

ax.legend()

plt.show()