import jittor as jt
from jittor import nn
import jittor.transform as transform
import numpy as np
import pickle
import tarfile
import matplotlib.pyplot as plt
from tqdm import tqdm

class MyDataset(jt.dataset.Dataset):
    def __init__(self, data, labels):
        super().__init__()
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

# ResNet模型
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
            )

    def execute(self, x):
        out = nn.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = nn.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def execute(self, x):
        out = nn.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.pool(out, 4, op='mean')
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        return out

# RNN模型
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.input_size = 96
        self.hidden_size = 128
        self.num_layers = 5
        self.output_size = 10
        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def execute(self, x):
        h0 = jt.zeros((self.num_layers, x.size(0), self.hidden_size))
        out, _ = self.rnn(x, h0)
        out = out.transpose(0, 1)
        out = self.fc(out[:, -1, :])
        return out

# 解压
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# 加载数据集
def load_data(tarfile_path):
    with tarfile.open(tarfile_path) as tar:
        tar.extractall()
        train_data = []
        train_labels = []
        for i in range(1, 6):
            batch_data = unpickle(f'cifar-10-batches-py/data_batch_{i}')
            train_data.extend(batch_data[b'data'].reshape(-1, 3, 32, 32))
            train_labels += batch_data[b'labels']
        test_data = unpickle('cifar-10-batches-py/test_batch')
        return train_data, train_labels, test_data[b'data'].reshape(-1, 3, 32, 32), test_data[b'labels']
    
# 训练CNN
def train_cnn(model, train_loader, test_loader, epoches=10, data_augmentation=False):

    criterion = nn.CrossEntropyLoss()
    optimizer = nn.Adam(model.parameters(), lr=1e-4)

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    # 数据增广
    transform_train = transform.Compose([
        transform.RandomHorizontalFlip(),
        transform.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.4),
        transform.RandomAffine(degrees=15),
        transform.RandomResizedCrop(size=(32, 32), scale=(0.5, 1.0)),
    ])

    for epoch in range(epoches):
        print('Epoch: %d' % (epoch + 1))

        train_loss = 0
        correct = 0
        total = 0
        model.train()

        for i, (inputs, labels) in tqdm(enumerate(train_loader), total=int(np.ceil(len(train_loader)/batch_size)), desc='Training'):
            # 数据增广
            if data_augmentation:
                tran_inputs = []
                for i in range(inputs.shape[0]):
                    img = inputs[i]
                    img = jt.transpose(img, (1, 2, 0))
                    img = np.array(transform_train(255-img))
                    
                    img = np.transpose(img, (2, 0, 1))
                    tran_inputs.append(img)
                inputs = jt.array(tran_inputs)

            # 训练
            outputs = model(inputs.float32())
            loss = criterion(outputs, labels)
            train_loss += np.squeeze(loss.numpy())
            optimizer.step(loss)

            predicted, _ = jt.argmax(outputs, 1)
            predicted = np.array(predicted)
            labels = np.squeeze(np.array(labels))
            correct += np.sum(predicted == labels)
            total += labels.shape[0]
        
        print('Train Loss: %.3f Train Accuracy: %d %%' % (train_loss / int(np.ceil(len(train_loader)/batch_size)), 100 * correct / total))
        train_losses.append(train_loss / int(np.ceil(len(train_loader)/batch_size)))
        train_accs.append(100 * correct / total)

        test_loss, test_acc = test_cnn(model, test_loader)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

    return model, train_losses, test_losses, train_accs, test_accs

# 测试CNN
def test_cnn(model, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    test_loss = 0

    with jt.no_grad():
        for i, (inputs, labels) in tqdm(enumerate(test_loader), total=int(np.ceil(len(test_loader)/batch_size)), desc='Testing'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += np.squeeze(loss.numpy())

            predicted, _ = jt.argmax(outputs.data, 1)
            predicted = np.array(predicted)
            labels = np.squeeze(np.array(labels))
            correct += np.sum(predicted == labels)

    print('Test Loss: %.3f Test Accuracy: %d %%' % (test_loss / int(np.ceil(len(test_loader)/batch_size)), 100 * correct / len(test_loader)))

    return test_loss / int(np.ceil(len(test_loader)/batch_size)), 100 * correct / len(test_loader)

# 训练RNN
def train_rnn(model, train_loader, test_loader, epoches=10, data_augmentation=False):

    criterion = nn.CrossEntropyLoss()
    optimizer = nn.Adam(model.parameters(), lr=1e-4)

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    # 数据增广
    transform_train = transform.Compose([
        transform.RandomHorizontalFlip(),
        transform.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.4),
        transform.RandomAffine(degrees=15),
        transform.RandomResizedCrop(size=(32, 32), scale=(0.5, 1.0)),
    ])

    for epoch in range(epoches):
        print('Epoch: %d' % (epoch + 1))

        train_loss = 0
        correct = 0
        total = 0
        model.train()

        for i, (inputs, labels) in tqdm(enumerate(train_loader), total=int(np.ceil(len(train_loader)/batch_size)), desc='Training'):
            # 数据增广
            if data_augmentation:
                tran_inputs = []
                for i in range(inputs.shape[0]):
                    img = inputs[i]
                    img = jt.transpose(img, (1, 2, 0))
                    img = np.array(transform_train(255-img))
                    
                    img = np.transpose(img, (2, 0, 1))
                    tran_inputs.append(img)
                inputs = jt.array(tran_inputs)

            # 训练
            inputs = inputs.view(inputs.size(0), -1, 32*3)
            outputs = model(inputs.float32())
            loss = criterion(outputs, labels)
            train_loss += np.squeeze(loss.numpy())
            optimizer.step(loss)

            predicted, _ = jt.argmax(outputs, 1)
            predicted = np.array(predicted)
            labels = np.squeeze(np.array(labels))
            correct += np.sum(predicted == labels)
            total += labels.shape[0]
            
        print('Train Loss: %.3f Train Accuracy: %d %%' % (train_loss / int(np.ceil(len(train_loader)/batch_size)), 100 * correct / total))
        train_losses.append(train_loss / int(np.ceil(len(train_loader)/batch_size)))
        train_accs.append(100 * correct / total)

        test_loss, test_acc = test_rnn(model, test_loader)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

    return model, train_losses, test_losses, train_accs, test_accs

# 测试RNN
def test_rnn(model, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    test_loss = 0

    with jt.no_grad():
        for i, (inputs, labels) in tqdm(enumerate(test_loader), total=int(np.ceil(len(test_loader)/batch_size)), desc='Testing'):
            inputs = inputs.view(inputs.size(0), -1, 32*3)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += np.squeeze(loss.numpy())

            predicted, _ = jt.argmax(outputs.data, 1)
            predicted = np.array(predicted)
            labels = np.squeeze(np.array(labels))
            correct += np.sum(predicted == labels)

    print('Test Loss: %.3f Test Accuracy: %d %%' % (test_loss / int(np.ceil(len(test_loader)/batch_size)), 100 * correct / len(test_loader)))

    return test_loss / int(np.ceil(len(test_loader)/batch_size)), 100 * correct / len(test_loader)

# 减少90%的标签0-4数据
def partition_data(train_data, train_labels):
    total_mask = np.zeros_like(train_labels) 
    
    for i in range(5):
        mask = (train_labels == i)
        indices = np.where(mask == 1)[0]
        indices = np.random.choice(indices, size=int(0.1 * len(indices)), replace=False)
        mask = np.zeros_like(mask)
        mask[indices] = 1
        total_mask += mask

    mask = (train_labels >= 5)
    total_mask += mask
    train_data = train_data[total_mask == 1]
    train_labels = train_labels[total_mask == 1]

    return train_data, train_labels

if __name__ == "__main__":
    partition = True
    model_name = "RNN"
    data_augmentation = False
    epoches = 50
    print("Use model:", model_name, "\nPartition data:", partition, "\nData augmentation:", data_augmentation, "\nEpoches:", epoches)

    # 导入CIFAR-10数据集
    tarfile_path = r"D:\Deep Learning and Its Application\Homework2\cifar-10-python.tar.gz"
    train_data, train_labels, test_data, test_labels = load_data(tarfile_path)

    train_data = jt.array(train_data).float32()
    train_labels = jt.array(train_labels).int64()
    test_data = jt.array(test_data).float32()
    test_labels = jt.array(test_labels).int64()

    # 减少90%的标签0-4训练数据
    if partition:
        train_data, train_labels = partition_data(train_data, train_labels)

    # 创建模型
    if model_name == "CNN":
        model = ResNet(BasicBlock, [2,2,2,2])
    elif model_name == "RNN":
        model = RNN()

    # 使用CUDA
    jt.flags.use_cuda = 1

    # 创建数据加载器
    train_dataset = MyDataset(train_data, train_labels)
    test_dataset = MyDataset(test_data, test_labels)
    batch_size = 256
    train_loader = jt.dataset.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = jt.dataset.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 训练模型
    if model_name == "CNN":
        model, train_losses, test_losses, train_accs, test_accs = train_cnn(model, train_loader, test_loader, epoches=epoches, data_augmentation=data_augmentation)
    elif model_name == "RNN":
        model, train_losses, test_losses, train_accs, test_accs = train_rnn(model, train_loader, test_loader, epoches=epoches, data_augmentation=data_augmentation)

    # 绘制Loss曲线
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.legend()
    plt.title('Loss')
    plt.show()

    # 绘制Accuracy曲线
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.show()