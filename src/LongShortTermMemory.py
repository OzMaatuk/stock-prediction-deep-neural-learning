from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping

# TODO:
# from tensorflow.keras.regularizers import l2

# # Add this inside your LSTM layers
# lstm_layer = LSTM(units, kernel_regularizer=l2(0.01), return_sequences=True)


# from tensorflow.keras.layers import BatchNormalization

# # Add this after your LSTM layers
# batch_norm_layer = BatchNormalization()


# from tensorflow.keras.callbacks import LearningRateScheduler

# # Define the scheduler function
# def scheduler(epoch, lr):
#     if epoch < 10:
#         return lr
#     else:
#         return lr * tf.math.exp(-0.1)

# # Add this to your callbacks when fitting the model
# lr_scheduler = LearningRateScheduler(scheduler)




class LongShortTermMemory:
    def __init__(self, project_folder):
        self.project_folder = project_folder

    def get_defined_metrics(self):
        defined_metrics = [
            MeanSquaredError(name='MSE')
        ]
        return defined_metrics

    def get_callback(self):
        callback = EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=1)
        return callback

    def create_model(self, x_train):
        model = Sequential()
        # 1st layer with Dropout regularisation
        # * units = add 100 neurons is the dimensionality of the output space
        # * return_sequences = True to stack LSTM layers so the next LSTM layer has a three-dimensional sequence input
        # * input_shape => Shape of the training dataset
        model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        # 20% of the layers will be dropped
        model.add(Dropout(0.2))
        # 2nd LSTM layer
        # * units = add 50 neurons is the dimensionality of the output space
        # * return_sequences = True to stack LSTM layers so the next LSTM layer has a three-dimensional sequence input
        model.add(LSTM(units=50, return_sequences=True))
        # 20% of the layers will be dropped
        model.add(Dropout(0.2))
        # 3rd LSTM layer
        # * units = add 50 neurons is the dimensionality of the output space
        # * return_sequences = True to stack LSTM layers so the next LSTM layer has a three-dimensional sequence input
        model.add(LSTM(units=50, return_sequences=True))
        # 50% of the layers will be dropped
        model.add(Dropout(0.5))
        # 4th LSTM layer
        # * units = add 50 neurons is the dimensionality of the output space
        model.add(LSTM(units=50))
        # 50% of the layers will be dropped
        model.add(Dropout(0.5))
        # Dense layer that specifies an output of one unit
        model.add(Dense(units=1))
        model.summary()
        #tf.keras.utils.plot_model(model, to_file=os.path.join(self.project_folder, 'model_lstm.png'), show_shapes=True,
        #                          show_layer_names=True)
        return model
