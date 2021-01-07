#Este codigo no ha podido ser agregado a la funcion de perdidad de Keras 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
from . import backend as K
from .utils.generic_utils import deserialize_keras_object
from .utils.generic_utils import serialize_keras_object


def RF(y_true, y_pred):
    discount = 0.9
    z = K.abs(y_true - y_pred)
    res = - (K.mean(K.log(z)*discount))

    return (res)
