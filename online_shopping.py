#%% import packages
import os 
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
print(keras.backend.backend())

#%%
from keras import layers,losses,metrics,activations,callbacks,initializers,regularizers,optimizers
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import sklearn.preprocessing
from sklearn import model_selection
import sklearn, datetime, pickle
import tensorflow as tf
from sklearn.metrics import accuracy_score,f1_score
import string
import json 
#%%
#reading csv file
CSV_PATH = os.path.join(os.getcwd(),'dataset','ecommerceDataset.csv')
df = pd.read_csv(CSV_PATH,names = ['category','description'],header=None)
print(df.head())
print(df.shape)
print(df.info())
df.category.value_counts()

#%% Checking missing values and duplicates
print("Total missing values in columns: ","\n",df.isna().sum())
print("Duplicates: ",df.duplicated().sum())

# %% print out missing values in description column
missing_values = df[df['description'].isna()]
print(missing_values)

# %%
#dropping the one single missing value row
df.dropna(inplace=True)
df.shape

# %%
#replacing name for clothing n accessories for standardization naming 
df.category.replace("Clothing & Accessories", "Clothing_Accessories", inplace=True)

# %%
#Checking total values
df.category.value_counts()

# %%
# Dropping Duplicates
df = df.drop_duplicates()
print(df)

# %%
# Data Cleaning (Punctuation,uneven cases,numbers)
df['description'] = df['description'].str.replace(f"[{string.punctuation}]","",regex=True)
df['description'] = df['description'].str.lower()
df['description'] = df['description'].str.replace(r"\d+",'',regex=True)
print(df['description'])

# %%
#preprocessing
features = df['description'].values
label = df['category'].values
# uncomment to confirm values
# print(features)
# print(label)

# Performing label encoding
label_encoder =  sklearn.preprocessing.LabelEncoder()
label_encoded = label_encoder.fit_transform(label)
print(label_encoded[:5])
#checking first 5 categories that is changed from string to numbers

#%%
#perform inverse transform with the encoder to convert back to original values
sample_categories = [0,1,2,3]
print(label_encoder.inverse_transform(sample_categories))

# %%
# Data Splitting
SEED = 42
x_train,x_split,y_train,y_split = model_selection.train_test_split(features,label_encoded,train_size=0.7
                                                                 ,random_state = SEED)
x_val,x_test,y_val,y_test = model_selection.train_test_split(x_split,y_split,train_size = 0.5, 
                                                             random_state = SEED)

# %%
#NLP
# (A) Tokenization
tokenizer = layers.TextVectorization(max_tokens=10000,output_sequence_length=200)
tokenizer.adapt(x_train)

# %%
# [Optional] test the tokenizer
sample_text = x_train[:2]
sample_tokens = tokenizer(sample_text)
print(sample_text)
print(sample_tokens)

# %%
# (B) Embedding 
embedding = layers.Embedding(10000,64)

# %%
# using bi directional lstm for ambiguity of data
model = keras.Sequential()
# (A) NLP layers
model.add(tokenizer)
model.add(embedding)
# (B) RNN (Wrap up with bi directional lstm layer)
model.add(layers.Bidirectional(layers.LSTM(32,return_sequences=False)))
# (C) Output layer
model.add(layers.Dense(len(df['category'].unique()),activation="softmax"))

# %%
# Directing the model to Tensorboard
log_dir = "logs" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# %%
# Model compile 
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# %%
# Model Training
history = model.fit(x_train,y_train,validation_data=(x_val,y_val),batch_size=32,epochs=5,callbacks=[tensorboard_callback])

# %%
# Check training result
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
plt.plot(history.epoch,history.history['loss'])
plt.plot(history.epoch,history.history['val_loss'])
plt.legend(['Training Loss', 'Validation Loss'])
plt.title('Loss graph')
plt.show()

#%%
# Evaluate the model with test data
print(model.evaluate(x_test,y_test))

#%%
y_pred = np.argmax(model.predict(x_test),axis=1)
print(y_pred)

#%%
#Results
accuracy = accuracy_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred,average='weighted')
accuracy_percentage = accuracy * 100
print(f"Accuracy:{accuracy_percentage:.2f}")
print(f"F1-Score: {f1:.2f}")

#%%
# Save the label encoder via pickle
with open('shop_encoder.pkl','wb') as f:
    pickle.dump(label_encoder,f)

#%%
os.makedirs("saved_models",exist_ok=True)
model.save("saved_models/models.h5")

vocab = tokenizer.get_vocabulary()
vocab_dict = {i: word for i, word in enumerate(vocab)}
with open ("saved_models/tokenizer.json","w") as f:
    json.dump(vocab_dict,f)