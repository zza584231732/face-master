import mxnet as mx
import numpy as np


model=mx.model.FeedForward.load('../my_model/mface-ext-model',3,num_batch_size=1)
internals=model.symbol.get_internals()
print (internals.list_outputs())
fea_symbol=internals['blockgrad0_output']
print fea_symbol


feature_extractor=mx.model.FeedForward(symbol=fea_symbol,
                                       numpy_batch_size=1,arg_params=model.arg_params,aux_params=model.aux_params,allow_extra_params=True)


feature_extractor.save('../my_model/feture-model',1)

new_model=mx.model.FeedForward.load('../my_model/feture-model',1,num_batch_size=1)
new_internals=new_model.symbol.get_internals()
print (new_internals.list_outputs())
vec = ['../my_model/feture-model', '0001']
_, arg_params, aux_params = mx.model.load_checkpoint(vec[0], int(vec[1]))
