# import the dataset 
from dataset.data import *

# import the USE model to apply as keras layer
from model.USE_model import *

# create the model with functional api
from keras import layers 

# provide the input layer 
inputs = layers.Input(shape=[], dtype=tf.string)

# Universal Sentence encoder (for vectorization and embedding layer)
USE_model = USEModel()(inputs)

# use of custom dense layers rather than LSTM (USE model outputs different shape)
custom_layer = layers.Dense(128, activation='relu')(USE_model)

custom_layer = layers.Dense(128, activation='relu')(custom_layer)

# return the predictions in the output layer
outputs = layers.Dense(len(class_names), activation='softmax')(custom_layer)


# join the layers 
from keras import Model
model2 = Model(inputs, outputs)


# see the model's architecture
model2.summary()


# compile the model 
from keras import losses, optimizers
model2.compile(
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
history2 = model2.fit(
    train_dataset, # training dataset
    epochs=10, # train for 10 epochs
    batch_size=128, # batches of 128
    steps_per_epoch=len(train_dataset), # number of batches for the whole dataset

    validation_data=test_dataset, # validation dataset
    validation_steps=len(test_dataset), # number of batches for the validation dataset

    callbacks=[early, reduceLr]
)

# save the model
model2.save('models/saved_models/model2/model2.keras')

# save the weigths
model2.save_weights('models/saved_models/model2/model2.weights.h5')

# evaluate the model 
evaluation2 = model2.evaluate(test_dataset)

# use in the terminal to include the emojis : $env:PYTHONUTF8 = "1"  

# provide accuracy score and loss 
print(f"Accuracy: {evaluation2[1]}\nLoss: {evaluation2[0]}")

from Functions.helperFunctions import * # import the helper Functions to plot the loss curves (see if model is overfitting, underfitting or performing well)
plot_loss_curves(history2) # plot loss curves
plt.show() # show the plot





