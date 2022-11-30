from .layer_tests import exec_layer_tests
from .data_tests import exec_data_tests

def exec_all_tests():
    exec_layer_tests()
    exec_data_tests()