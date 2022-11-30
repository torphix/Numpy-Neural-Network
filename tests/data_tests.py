from src.data import MNISTDataloader

def test_dataloader():
    dataloader = MNISTDataloader([0.8,0.1,0.1], shuffle=True, n_samples=5)



def exec_data_tests():
    test_dataloader()