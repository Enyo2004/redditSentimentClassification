# import the dataset 
from dataset.data import *

# import the extra layers 
from model.extra_layers import *

# use functional api 
from keras import layers 

# get the inputs (text)
inputs = layers.Input(shape=(1,), dtype=tf.string)

# vectorize the texts 
vectorization_layer = vectorization(inputs)

# make the embedding matrix
embedding_layer = embedding(vectorization_layer)

# use Bidirectional LSTM 
LSTM_bidirectional = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(embedding_layer) # return sequences to meet the tensor shape in globalMaxPooling 

# use of globalMaxPooling to find key words in the text
pooling = layers.GlobalMaxPooling1D()(LSTM_bidirectional)

# outputs (use softmax for multiclass classification)
outputs = layers.Dense(len(class_names), activation='softmax')(pooling)

# join the layers
from keras import Model
model1 = Model(inputs, outputs)

# see the model's architecture
#model1.summary()


# compile the model 
from keras import losses, optimizers
model1.compile(
    loss=losses.sparse_categorical_crossentropy, # use sparse for int categorical data
    optimizer=optimizers.Adam(), # use of ADAM (Adaptive Moment estimator)
    metrics=['accuracy'] # accuracy as metric
)


# create callbacks for extra add-ons when training 
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

early = EarlyStopping( # stop early if the model does not have better metrics when training
    monitor='val_accuracy',
    patience=4,
    restore_best_weights=True, # restore the best weights the model got at a certaing epoch
    verbose=0
)

reduceLr = ReduceLROnPlateau( # reduce the learning rate to avoid model overshooting in the training
    monitor='val_accuracy',
    patience=2,
    verbose=0,
)

# train the model 
history1 = model1.fit(
    train_dataset, # training dataset
    epochs=10, # train for 10 epochs
    batch_size=128, # batches of 128
    steps_per_epoch=len(train_dataset), # number of batches for the whole dataset

    validation_data=test_dataset, # validation dataset
    validation_steps=len(test_dataset), # number of batches for the validation dataset

    callbacks=[early, reduceLr]
)

# save the model
model1.save('models/saved_models/model1/model1.keras')

# save the weigths
model1.save_weights('models/saved_models/model1/model1.weights.h5')

# evaluate the model 
evaluation1 = model1.evaluate(test_dataset)

# use in the terminal to include the emojis : $env:PYTHONUTF8 = "1"  

# provide accuracy score and loss 
print(f"Accuracy: {evaluation1[1]}\nLoss: {evaluation1[0]}")

from Functions.helperFunctions import * # import the helper Functions to plot the loss curves (see if model is overfitting, underfitting or performing well)
plot_loss_curves(history1) # plot loss curves
plt.show() # show the plot



