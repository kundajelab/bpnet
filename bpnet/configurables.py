"""Other gin configurables
"""
from gin import config
import keras

# keras.optimizers
config.external_configurable(keras.optimizers.Adam, module='keras.optimizers')
config.external_configurable(keras.optimizers.RMSprop, module='keras.optimizers')
config.external_configurable(keras.optimizers.Adagrad, module='keras.optimizers')
config.external_configurable(keras.optimizers.Adadelta, module='keras.optimizers')
config.external_configurable(keras.optimizers.Adamax, module='keras.optimizers')
config.external_configurable(keras.optimizers.Nadam, module='keras.optimizers')
config.external_configurable(keras.optimizers.SGD, module='keras.optimizers')
