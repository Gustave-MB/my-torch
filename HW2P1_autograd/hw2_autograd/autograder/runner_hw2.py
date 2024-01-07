import numpy as np

# NOTE: If you are on Windows and are having trouble with imports, try to run
# this file from inside the autograder directory.
import sys

sys.path.append('./..')
sys.path.append('mytorch')
import mytorch
from test_conv_functional import *
from test_conv import *

version = "1.0.2"

tests = [
    {
        'name': '1.1 - Functional Backward - Conv1d',
        'autolab': 'Functional Backward - Conv1d',
        'handler': test_conv1d_backward,
        'value': 5,
    },
    {
        'name': '1.2 - Functional Backward - Conv2d',
        'autolab': 'Functional Backward - Conv2d',
        'handler': test_conv2d_backward,
        'value': 5,
    },
    {
        'name': '2.1 - Conv1d (Autograd) Forward',
        'autolab': 'Conv1d (Autograd) Forward',
        'handler': test_cnn1d_layer_forward,
        'value': 2,
    },
    {
        'name': '2.2 - Conv1d (Autograd) Backward',
        'autolab': 'Conv1d (Autograd) Backward',
        'handler': test_cnn1d_layer_backward,
        'value': 3,
    },
    {
        'name': '2.3 - Conv2d (Autograd) Forward',
        'autolab': 'Conv2d (Autograd) Forward',
        'handler': test_cnn2d_layer_forward,
        'value': 2,
    },
    {
        'name': '2.4 - Conv2d (Autograd) Backward',
        'autolab': 'Conv2d (Autograd) Backward',
        'handler': test_cnn2d_layer_backward,
        'value': 3,
    },
]

if __name__ == '__main__':
    print("Autograder version {}\n".format(version))
    run_tests(tests)
