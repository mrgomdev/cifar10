from typing import Tuple, Optional, Callable
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


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir_path: str, transform: Optional[Callable[[np.ndarray], torch.Tensor]] = torchvision.transforms.ToTensor(), target_transform: Optional[Callable[[int], torch.Tensor]] = None):
        super(TrainDataset, self).__init__()
        self.data_dir_path = data_dir_path
        self.images = []
        self.labels = []

        to_pil = torchvision.transforms.ToPILImage(mode='RGB')

        for path in glob.glob(os.path.join(self.data_dir_path, 'data_batch_*')):
            with open(path, 'rb') as file:
                batch = pickle.load(file, encoding='bytes')
                each_data = einops.rearrange(batch[b'data'], 'batch (channel height width) -> batch height width channel', channel=3, height=32, width=32)
                each_labels = np.asarray(batch[b'labels'], dtype=np.int64)
                self.images.extend(map(to_pil, each_data))
                self.labels.extend(list(each_labels))
        assert len(self.images) == len(self.labels)

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        image, label = self.images[index], self.labels[index]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.images)

torchvision.datasets.CIFAR10

# Dataload
cifar_dir_path = './data/cifar-10-batches-py'
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = TrainDataset(cifar_dir_path, transform=transform)
train_dataset = torch.utils.data.Subset(train_dataset, indices=range(1024))
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)

test_dataset = torch.utils.data.Subset(train_dataset, indices=range(1024))
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

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

classifier = Classifier()
classifier = classifier.cuda()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

# Train
def train(classifier: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader):
    old_training = classifier.training

    classifier.train()
    for epoch in tqdm(range(10)):
        for i, (images, labels) in enumerate(train_dataloader):
            images = images.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch: {epoch}, Step: {i}, Loss: {loss.item()}')

    classifier.train(mode=old_training)


# Test
def eval(classifier: torch.nn.Module, test_dataloader: torch.utils.data.DataLoader):
    old_training = classifier.training

    classifier.eval()
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.cuda()
            labels = labels.cuda()

            outputs = classifier(images)
            predicted = torch.max(outputs, 1).indices
            print(f"Predicted: {predicted}, Labels: {labels}")
            print(f"Accuracy: {sum(predicted == labels) / len(labels)}")

    classifier.train(mode=old_training)

eval(classifier=classifier, test_dataloader=test_dataloader)
train(classifier=classifier, train_dataloader=train_dataloader)
eval(classifier=classifier, test_dataloader=test_dataloader)
