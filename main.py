import keras
from keras.models import Model

from models import sample_dense_model
from models import sample_conv_model

model_type = 'conv'

if model_type == 'conv':
    input_tensor, output_tensor = sample_conv_model(*args, **kwargs)
    model = Model(input_tensor, output_tensor)
elif model_type == 'dense':
    input_tensor, output_tensor = sample_dense_model(*args, **kwargs)
    model = Model(input_tensor, output_tensor)
else:
    warnings.warn('No model type implemented for {}.'.format(model_type))

model.compile(*args, **kwargs)

model.fit(*args, **kwargs)

model.save_weights(*args, **kwargs)
