import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys
import pickle
from keras.layers import Input
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.layers import Reshape, Permute, multiply
from keras.layers import TimeDistributed, Dense, Dropout
from keras.layers import GRU, Bidirectional
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from keras import optimizers
from tensorflow.python.client import device_lib
from sklearn.metrics import roc_curve, auc
from keras import backend as K
import datetime
from keras.models import load_model

# adapted from https://github.com/Jack-0-0/Cross-Species-Pre-training/blob/master/train/cross_species_train.py
########################
####### Settings ####### 
########################

# set path for training data
train = np.load('/import/c4dm-datasets/manxShearwaters/adult_vs_chick/train_PCEN.npy', allow_pickle=True)
# set path for validation data
val = np.load('/import/c4dm-datasets/manxShearwaters/adult_vs_chick/val_PCEN.npy', allow_pickle=True)
# set path for test data
test = np.load('/import/c4dm-datasets/manxShearwaters/adult_vs_chick/test_PCEN.npy', allow_pickle=True)



# # set path for pretrained model
# pre_tr_path = 'laughter_model_PCEN.h5'

# choose which GPU to use
gpu = "0"

# set number of mels
# all pretrained and baseline networks are at 8,000 Hz so keep mels set to 45
mels = 45

# set model type
# set to 'baseline' to train model without any pretraining ( from scratch)
# set to 'frozen' to train model with pretraining but all layers frozen except last layer
# set to 'unfrozen' to train model with pretraining with all layers unfrozen
model_type = 'baseline'  


# Check data normalization and scaling!!

# # set whether scaling should be used
# # only set to True if using pretraining and not using PCEN
# scaling = False

# # set paths to laughter datasets, this data will be used to scale the hyena data if necessary
# if scaling:
#     train_ch = np.load('ch_train_6_6_64_ds.npy')
#     train_de = np.load('de_train_6_6_64_ds.npy')
#     train_en = np.load('en_train_6_6_64_ds.npy')


# set save folder identifier
# save folder will be named in format saved_models/date_time_filename_sfid 
sfid = 'baseline_PCEN' 

# set path for tensorboard logging:
logdir = 'logs'
checkpoints = 'checkpoints'


#classification task:
# adult grunts, adult bouts, chicks
nb_classes = 3
########################


# def create_pretrain_model(freeze_layers, pre_trained_model_path):
#     """Load pretrained model and unfreeze all or the last layer.
#         # Arguments
#         freeze_layers: Boolean, whether to freeze layers. Output layer always remains trainable.
#         pre_trained_model_path: path for pre trained model.

#     # Returns
#         pretrained Keras model which can then be compiled and fit.
#     """    
#     laughter_model = load_model(pre_trained_model_path) # load parent model
#     p_model = Model(laughter_model.inputs, laughter_model.layers[-2].output) # remove last layer
#     x = p_model.output
#     x = Dense(8, activation='sigmoid', name='hyena_on_laughter')(x) # add last layer in hyena data format
#     model = Model(inputs=p_model.input, outputs=x)
#     if freeze_layers:
#         for layer in p_model.layers:
#             layer.trainable = False
#     return model


def create_baseline_model(filters, gru_units, dropout, bias, mels, nb_classes):
    """Create baseline convolutional recurrent model.

    # Arguments
        filters: number of filters in the convolutional layers.
        gru_units: number of gru units in the GRU layers.
        dropout: neurons to drop out during training. Values of between 0 to 1.
        bias: set to True or False. Should be False when using BatchNorm, True when not.

    # Returns
        Keras functional model which can then be compiled and fit.
    """
    inp = Input(shape=(259, mels, 1))
    x = Conv2D(filters, (3,3), padding='same', activation='relu', use_bias=bias)(inp)
    x = MaxPooling2D(pool_size=(1,5))(x)
    x = Conv2D(filters, (3,3), padding='same', activation='relu', use_bias=bias)(x)
    x = MaxPooling2D(pool_size=(1,2))(x)
    x = Conv2D(filters, (3,3), padding='same', activation='relu', use_bias=bias)(x)
    x = MaxPooling2D(pool_size=(1,2))(x)

    x = Reshape((x_train.shape[-3], -1))(x)

    x = Bidirectional(GRU(units=gru_units, activation='tanh', dropout=dropout, 
                              recurrent_dropout=dropout, return_sequences=bias), merge_mode='mul')(x)
    
    x = TimeDistributed(Dense(512, activation='relu', use_bias=bias))(x)
    x = Dropout(rate=dropout)(x)
    x = TimeDistributed(Dense(256, activation='relu', use_bias=bias))(x)
    x = Dropout(rate=dropout)(x)
    x = TimeDistributed(Dense(128, activation='relu', use_bias=bias))(x)
    x = Dropout(rate=dropout)(x)
    output = Dense(nb_classes, activation='sigmoid')(x)
    model = Model(inputs=[inp], outputs=output)
    return model


def save_folder(date_time, sfid, logs_folder, checkpoints_folder):
    """Create save folder and return the path.

    # Arguments
        date_time: Current time as per datetime.datetime.now()

    # Creates
        directory at save_folder location, if it does not exist already.

    # Returns
        path to save folder.
    """
    date_now = str(date_time.date())
    time_now = str(date_time.time())
    sf = "saved_models/" + date_now + "_" + time_now + "_" \
    + os.path.basename(__file__).split('.')[0] + '_' + sfid
    if not os.path.isdir(sf):
        os.makedirs(sf)

    lf = sf +'/' + logs_folder
    if not os.path.isdir(lf):
        os.makedirs(lf)
    chkf = sf +'/' +checkpoints_folder
    if not os.path.isdir(chkf):
        os.makedirs(chkf)


    return sf, lf, chkf

        
def save_model(save_folder):
    """Saves model and history file.

    # Arguments
        save_folder: path for directory to save model and related history and metrics.

    # Output
        saves model and history.
    """ 
    model.save(save_folder + '/savedmodel' + '.h5')
    with open(save_folder +'/history.pickle', 'wb') as f_save:
        pickle.dump(model_fit.history, f_save)


def plot_accuracy(model_fit, save_folder):
    """Plot the accuracy during training for the train and val datasets.

    # Arguments
        model_fit: model after training.
        save_folder: path for directory to save model and related history and metrics.

    # Output
        saves plots of train vs validation accuracy at each epoch.
    """ 
    train_acc = model_fit.history['binary_accuracy']
    val_acc = model_fit.history['val_binary_accuracy']
    epoch_axis = np.arange(1, len(train_acc) + 1)
    plt.title('Train vs Validation Accuracy')
    plt.plot(epoch_axis, train_acc, 'b', label='Train Acc')
    plt.plot(epoch_axis, val_acc,'r', label='Val Acc')
    plt.xlim([1, len(train_acc)])
    plt.xticks(np.arange(min(epoch_axis), max(epoch_axis) + 1, round((len(train_acc) / 10) + 0.5)))
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.savefig(save_folder + '/accuracy.png')
    plt.show()
    plt.close()
    

def plot_loss(model_fit, save_folder):
    """Plot the loss during training for the train and val datasets.

    # Arguments
        model_fit: model after training.
        save_folder: path for directory to save model and related history and metrics.

    # Output
        saves plots of train vs validation loss at each epoch.
    """ 
    train_loss = model_fit.history['loss']
    val_loss = model_fit.history['val_loss']
    epoch_axis = np.arange(1, len(train_loss) + 1)
    plt.title('Train vs Validation Loss')
    plt.plot(epoch_axis, train_loss, 'b', label='Train Loss')
    plt.plot(epoch_axis, val_loss,'r', label='Val Loss')
    plt.xlim([1, len(train_loss)])
    plt.xticks(np.arange(min(epoch_axis), max(epoch_axis) + 1, round((len(train_loss) / 10) + 0.5)))
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.savefig(save_folder + '/loss.png')
    plt.show()
    plt.close()


def plot_ROC(model, x_test, y_test, save_folder):
    """Plot and save the ROC with AUC value.
    
    # Arguments
        model: model after training.
        x_test: inputs to the network for testing.
        y_test: actual outputs for testing.
        save_folder: path for directory to save model and related history and metrics.

    # Output
        saves plots of ROC.
    """ 
    predicted = model.predict(x_test).ravel()
    actual = y_test.ravel()
    fpr, tpr, thresholds = roc_curve(actual, predicted, pos_label=None)
    roc_auc = auc(fpr, tpr)
    plt.title('Test ROC AUC')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(save_folder + '/ROC.png')
    plt.show()
    plt.close()


def metrics(x, y, save_folder, threshold, ds_name):
    """Calculate the TPR, TNR, FPR, FNR and F1 score.

    # Arguments
        x: inputs to the network.
        y: actual outputs.
        save_folder: path for directory to save model and related history and metrics.
        threshold: values greater than threshold get set to 1, values less than or
                   equal to the threshold get set to 0.
        df_name: identifier for text file.

    # Output
        saves True Positive Rate, True Negative Rate, False Positive Rate, False Negative Rate
        dependent on threshold.
    """
    predicted = model.predict(x)
    predicted[predicted > threshold] = 1
    predicted[predicted <= threshold] = 0
    actual = y
    TP = np.sum(np.logical_and(predicted == 1, actual == 1))
    FN = np.sum(np.logical_and(predicted == 0, actual == 1))
    TN = np.sum(np.logical_and(predicted == 0, actual == 0))
    FP = np.sum(np.logical_and(predicted == 1, actual == 0))
    TPR  = TP / (TP + FN + 1e-8)
    TNR  = TN / (TN + FP + 1e-8)
    FPR = FP / (FP + TN + 1e-8)
    FNR = FN / (FN + TP + 1e-8)
    precision = TP / (TP + FP + 1e-8)
    recall = TPR
    F1 = 2 * precision * recall / (precision + recall + 1e-8)
    metrics_dict = {'TPR': np.round(TPR, 3),
                    'TNR': np.round(TNR, 3),
                    'FPR' : np.round(FPR, 3),
                    'FNR' : np.round(FNR, 3),
                    'F1 Score' : np.round(F1, 3)
                   }
    with open(save_folder + '/' + ds_name + '_metrics.txt', 'w') as f:
        f.write(str(metrics_dict))


def save_arch(model, save_folder):
    """Saves the network architecture as a .txt file.

    # Arguments
        model: model after training.
        save_folder: path for directory to save model and related history and metrics.

    # Output
        saves network architecture.
    """
    with open(save_folder + '/architecture.txt','w') as a_save:
        model.summary(print_fn=lambda x: a_save.write(x + '\n'))


def reformat(dataset):
    """Reformat data into a suitable format.
    
    # Arguments
        dataset: dataset in format (id, spectro, label)
        
    # Returns
        x: spectros normalised across each mel band in format (n, timesteps, mel bands, 1)
        y: labels in format (n, timesteps, 8)
    """
    x = dataset[:, 1] 
    x = np.stack(x) # reshape to (n, mel bands, timesteps)
    x = np.expand_dims(np.moveaxis(x, 1, -1), axis=3) # reformat x to (n, timesteps, mel bands, 1)  
    y = dataset[:, 2] 
    y = np.moveaxis(np.stack(y), 1, -1) # reformat y to (n, timesteps, 8)
    return x, y


def scale(original_train, new_train):
    """Find scaling value.
    
    # Arguments
        original: data that was used to originally train the network.
        new_train: data that will be used to fine tune network.
    
    # Returns
        scaling value.
    """
    # find magnitude original training data
    o_mag = np.linalg.norm(np.stack(original_train[:,1]))
    # find magnitude new data
    n_mag = np.linalg.norm(np.stack(new_train[:,1]))
    # scale new data
    scale = o_mag / n_mag
    return scale


# GPU Setup
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
config = tf.ConfigProto()
tf.Session(config=config)

# scale data
# if scaling:
#     human_pretrain = np.vstack((train_ch, train_de))
#     human_pretrain = np.vstack((human_pretrain, train_en))
#     scale = scale(human_pretrain, train)
#     train[:, 1] = train[:, 1] * scale
#     val[:, 1] = val[:, 1] * scale
#     test[:, 1] = test[:, 1] * scale

# reformat datasets
x_train, y_train = reformat(train)
x_val, y_val = reformat(val)
x_test, y_test = reformat(test)

# reduce number of mel bands for spectrograms, to represent lower sample rate
x_train = x_train[:, :, :mels, :]
x_val = x_val[:, :, :mels, :]
x_test = x_test[:, :, :mels, :]

# create network
if model_type == 'baseline':
    model = create_baseline_model(filters=128, gru_units=128, dropout=0.5, bias=True, mels=mels, nb_classes=nb_classes)
# if model_type == 'frozen':
#     model = create_pretrain_model(freeze_layers=True, pre_trained_model_path=pre_tr_path)
# if model_type == 'unfrozen':
#     model = create_pretrain_model(freeze_layers=False, pre_trained_model_path=pre_tr_path)
print(model.summary())
adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['binary_accuracy'])

#create output folder:
date_time = datetime.datetime.now()
sf, lf, chkf= save_folder(date_time, sfid, logdir, checkpoints)

tensorboard_logging= TensorBoard(log_dir=lf)
#check if more metrics need to be explicitly added to tensorboard!
filepath = chkf + "/weights_{epoch:02d}_{val_loss:.2f}.hdf5"
save_checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=5)
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=1)
reduce_lr_plat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, verbose=1,
                                   mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.000001)


# train network
model_fit = model.fit(x_train, y_train, epochs=2500, batch_size=64,
                      validation_data=(x_val, y_val), callbacks=[early_stopping, reduce_lr_plat, tensorboard_logging, save_checkpoint])


# save network, training info, performance measures
save_model(sf)
plot_accuracy(model_fit, sf)
plot_loss(model_fit, sf)
plot_ROC(model, x_test, y_test, sf)
save_arch(model, sf)
metrics(x_test, y_test, sf, 0.5, 'test')
metrics(x_val, y_val, sf, 0.5, 'val')
