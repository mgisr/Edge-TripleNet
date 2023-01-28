from utils.dataset import EdgeTrainDataset, EdgeTestDataset, LightEdgeDataset
import torch


def test_dataset():
    train_data, test_data = EdgeTrainDataset(), EdgeTestDataset()

    assert len(train_data) == 60000
    assert len(test_data) == 10000

    X, Y, z = train_data[0]
    assert X.shape == (28, 28)
    assert Y.shape == (28, 28)
    assert z.shape == ()

    X, Y, z = test_data[0]
    assert X.shape == (28, 28)
    assert Y.shape == (28, 28)
    assert z.shape == ()


def test_dataloader():
    dataset = LightEdgeDataset()

    dataset.setup('fit')
    train_dataloader = dataset.train_dataloader()
    X, Y, z = next(iter(train_dataloader))
    assert X.shape == torch.Size([dataset.batch_size, 28, 28])
    assert Y.shape == torch.Size([dataset.batch_size, 28, 28])
    assert z.shape == torch.Size([dataset.batch_size])
