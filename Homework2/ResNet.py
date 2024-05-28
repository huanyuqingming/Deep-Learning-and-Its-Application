import jittor as jt
from jittor import nn
import numpy as np
import pickle
import tarfile
import matplotlib.pyplot as plt
from tqdm import tqdm
import jittor as jt
from jittor import nn

class MyDataset(jt.dataset.Dataset):
    def __init__(self, data, labels):
        super().__init__()
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

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

# 显示数据集图片
def show_image(img, label):
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    img = img.transpose((1, 2, 0))
    plt.imshow(img)
    plt.title(classes[label])
    plt.show()

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
def train_cnn(model, train_loader, test_loader):

    criterion = nn.CrossEntropyLoss()
    optimizer = nn.Adam(model.parameters(), lr=1e-4)

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    for epoch in range(50):
        print('Epoch: %d' % (epoch + 1))

        train_loss = 0
        correct = 0
        total = 0
        model.train()

        for i, (inputs, labels) in tqdm(enumerate(train_loader), total=int(np.ceil(len(train_loader)/batch_size)), desc='Training'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            train_loss += loss.numpy()
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
            test_loss += loss.numpy()

            predicted, _ = jt.argmax(outputs.data, 1)
            predicted = np.array(predicted)
            labels = np.squeeze(np.array(labels))
            correct += np.sum(predicted == labels)

    print('Test Loss: %.3f Test Accuracy: %d %%' % (test_loss / int(np.ceil(len(test_loader)/batch_size)), 100 * correct / len(test_loader)))

    return test_loss / int(np.ceil(len(test_loader)/batch_size)), 100 * correct / len(test_loader)

# 导入CIFAR-10数据集
tarfile_path = r"D:\Deep Learning and Its Application\Homework2\cifar-10-python.tar.gz"
train_data, train_labels, test_data, test_labels = load_data(tarfile_path)

train_data = jt.array(train_data).float32()
train_labels = jt.array(train_labels).int64()
test_data = jt.array(test_data).float32()
test_labels = jt.array(test_labels).int64()

# 创建模型
model = ResNet(BasicBlock, [2,2,2,2])

# 使用CUDA
jt.flags.use_cuda = 1

# 创建数据加载器
train_dataset = MyDataset(train_data, train_labels)
test_dataset = MyDataset(test_data, test_labels)
batch_size = 256
train_loader = jt.dataset.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = jt.dataset.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 训练模型
model, train_losses, test_losses, train_accs, test_accs = train_cnn(model, train_loader, test_loader)

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