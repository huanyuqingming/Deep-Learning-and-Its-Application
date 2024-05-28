import jittor as jt
from jittor import nn
import numpy as np
import pickle
import tarfile
import matplotlib.pyplot as plt
from tqdm import tqdm
import jittor as jt
from jittor import nn
import jittor.transform as transform

class MyDataset(jt.dataset.Dataset):
    def __init__(self, data, labels):
        super().__init__()
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

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
    
# 训练RNN
def train_rnn(model, train_loader, test_loader, labels_weight=False):
    optimizer = nn.Adam(model.parameters(), lr=1e-4)

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    for epoch in range(100):
        print('Epoch: %d' % (epoch + 1))

        train_loss = 0
        correct = 0
        total = 0
        model.train()

        for i, (inputs, labels) in tqdm(enumerate(train_loader), total=int(np.ceil(len(train_loader)/batch_size)), desc='Training'):
            # 类别权重
            if labels_weight:
                weights = [10 if label in [0, 1, 2, 3, 4] else 1 for label in labels]
                criterion = nn.CrossEntropyLoss(weight=jt.array(weights))
            else:
                criterion = nn.CrossEntropyLoss()

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

# 导入CIFAR-10数据集
tarfile_path = r"D:\Deep Learning and Its Application\Homework2\cifar-10-python.tar.gz"
train_data, train_labels, test_data, test_labels = load_data(tarfile_path)

train_data = jt.array(train_data).float32()
train_labels = jt.array(train_labels).int64()
test_data = jt.array(test_data).float32()
test_labels = jt.array(test_labels).int64()

# 减少90%的标签0-4训练数据
train_data, train_labels = partition_data(train_data, train_labels)

# 创建模型
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
model, train_losses, test_losses, train_accs, test_accs = train_rnn(model, train_loader, test_loader, labels_weight=False)

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
