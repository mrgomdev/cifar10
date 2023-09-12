from typing import Tuple, Optional, Callable, List
import os
import glob
import enum
import pickle
from tqdm import tqdm

import einops

from PIL import Image
import numpy as np
import torch.utils.data
import torchvision.transforms.functional


class Cifar10Label(enum.Enum):
    airplane = 0
    automobile = 1
    bird = 2
    cat = 3
    deer = 4
    dog = 5
    frog = 6
    horse = 7
    ship = 8
    truck = 9

    @classmethod
    def from_label(cls, label: int) -> 'Cifar10Label':
        return cls(label)

    @classmethod
    def from_name(cls, name: str) -> 'Cifar10Label':
        return cls[name]


class ImagesDataset(torch.utils.data.Dataset):
    def __init__(self, images: List[Image.Image], transform: Optional[Callable[[Image.Image], torch.Tensor]] = torchvision.transforms.ToTensor()):
        super(ImagesDataset, self).__init__()
        self.images = images
        self.transform = transform

    def __getitem__(self, index: int) -> np.ndarray:
        image = self.images[index]
        if self.transform:
            image = self.transform(image)

        return image

    def __len__(self) -> int:
        return len(self.images)


class Cifar10Dataset(ImagesDataset):
    def __init__(self, data_paths: List[str], transform: Optional[Callable[[np.ndarray], torch.Tensor]] = torchvision.transforms.ToTensor(), target_transform: Optional[Callable[[int], torch.Tensor]] = None):
        self.data_paths = list(data_paths)
        images = []
        labels = []

        to_pil = torchvision.transforms.ToPILImage(mode='RGB')

        for path in self.data_paths:
            with open(path, 'rb') as file:
                batch = pickle.load(file, encoding='bytes')
                each_data = einops.rearrange(batch[b'data'], 'batch (channel height width) -> batch height width channel', channel=3, height=32, width=32)
                each_labels = np.asarray(batch[b'labels'], dtype=np.int64)
                images.extend(map(to_pil, each_data))
                labels.extend(list(each_labels))
        assert len(images) == len(labels)

        super(Cifar10Dataset, self).__init__(images=images, transform=transform)
        self.target_transform = target_transform
        self.labels = labels

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        image, label = self.images[index], self.labels[index]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    @property
    def images_dataset(self) -> ImagesDataset:
        return ImagesDataset(images=self.images, transform=self.transform)


class TrainDataset(Cifar10Dataset):
    def __init__(self, data_dir_path: str, transform: Optional[Callable[[np.ndarray], torch.Tensor]] = torchvision.transforms.ToTensor(), target_transform: Optional[Callable[[int], torch.Tensor]] = None):
        paths = sorted(glob.glob(os.path.join(data_dir_path, 'data_batch_*')))
        super(TrainDataset, self).__init__(data_paths=paths, transform=transform, target_transform=target_transform)
        self.data_dir_path = data_dir_path


class TestDataset(Cifar10Dataset):
    def __init__(self, data_dir_path: str, transform: Optional[Callable[[np.ndarray], torch.Tensor]] = torchvision.transforms.ToTensor(), target_transform: Optional[Callable[[int], torch.Tensor]] = None):
        paths = sorted(glob.glob(os.path.join(data_dir_path, 'test_batch')))
        super(TestDataset, self).__init__(data_paths=paths, transform=transform, target_transform=target_transform)
        self.data_dir_path = data_dir_path

if __name__ == '__main__':
    # Dataload
    cifar_dir_path = './data/cifar-10-batches-py'
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = TrainDataset(cifar_dir_path, transform=transform)
    # train_dataset = torch.utils.data.Subset(train_dataset, indices=range(1024))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)

    test_dataset = TestDataset(cifar_dir_path, transform=transform)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    # Check
    # image, label = train_dataset[0]
    # Image.fromarray(np.transpose(image, (1, 2, 0))).show()

class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.top = torch.nn.Linear(32 * 32 * 32, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.flatten(x, 1)
        x = self.top(x)
        assert x.shape == (len(x), 10)

        return x

# Train
def train(classifier: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader, epochs: int):
    old_training = classifier.training

    classifier.train()
    # progress_bar = tqdm(range(epochs), position=1)
    progress_bar = range(epochs)
    for epoch in progress_bar:
        for i, (images, labels) in enumerate(train_dataloader):
            images = images.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # progress_bar.set_description(f'Epoch: {epoch}, Step: {i}, Loss: {loss.item()}', refresh=True)

    classifier.train(mode=old_training)


def predict(classifier: torch.nn.Module, dataset: ImagesDataset, batch_size: int) -> torch.Tensor:
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    outputs = []

    old_training = classifier.training
    classifier.eval()
    with torch.no_grad():
        for images_batch in dataloader:
            if isinstance(images_batch, tuple):
                images_batch = images_batch[0]
            images_batch = images_batch.cuda()
            outputs.extend(classifier(images_batch).cpu())

    classifier.train(mode=old_training)

    outputs = torch.stack(outputs)
    assert outputs.shape == (len(dataset), 10)

    return outputs


# Test
def eval(classifier: torch.nn.Module, test_dataloader: torch.utils.data.DataLoader) -> float:
    old_training = classifier.training

    labels = []
    predicteds = []

    classifier.eval()
    with torch.no_grad():
        for images_batch, labels_batch in test_dataloader:
            images_batch = images_batch.cuda()
            labels_batch = labels_batch.cuda()

            outputs = classifier(images_batch)
            predicted = torch.max(outputs, 1).indices

            labels.extend(labels_batch.cpu().numpy())
            predicteds.extend(predicted.cpu().numpy())

    classifier.train(mode=old_training)

    labels = np.asarray(labels)
    predicteds = np.asarray(predicteds)
    accuracy = np.sum(labels == predicteds) / len(labels)

    return accuracy

# Confusion matrix
def get_confusion_matrix(labels: np.ndarray, predicteds: np.ndarray) -> np.ndarray:
    assert labels.shape == predicteds.shape
    assert len(labels.shape) == 1

    matrix = np.zeros((10, 10), dtype=np.int64)
    for label, predicted in zip(labels, predicteds):
        matrix[label, predicted] += 1

    return matrix


if __name__ == '__main__':
    classifier = Classifier()
    classifier = classifier.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    print(f"Accuracy: {eval(classifier=classifier, test_dataloader=test_dataloader)}")
    for idx in tqdm(range(10), position=0):
        train(classifier=classifier, train_dataloader=train_dataloader, epochs=1)
        print(f"Accuracy: {eval(classifier=classifier, test_dataloader=test_dataloader)}")

        if False:
            test_dataset = torch.utils.data.Subset(test_dataset, indices=range(4))
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

            test_batch = next(iter(test_dataloader))
            predicteds = predict(classifier=classifier, dataset=ImagesDataset(images=test_batch[0], transform=None), batch_size=128)
            predicteds = torch.max(predicteds, 1).indices.cpu().numpy()
            labels = np.asarray(test_batch[1])
        else:
            predicteds = predict(classifier=classifier, dataset=test_dataset.images_dataset, batch_size=128)
            predicteds = torch.max(predicteds, 1).indices.cpu().numpy()
            labels = np.asarray(test_dataset.labels)
        confusion_matrix = get_confusion_matrix(labels=labels, predicteds=predicteds)
        torch.save(confusion_matrix, f'./confusion_matrix_{idx:02}.pt')
