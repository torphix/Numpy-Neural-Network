from .data_tests import exec_data_tests
from .layer_tests import exec_layer_tests
from .activation_tests import exec_activation_tests


def exec_all_tests():
    exec_data_tests()
    exec_layer_tests()
    exec_activation_tests()
