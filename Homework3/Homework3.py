import jittor as jt
from jittor import nn
import numpy as np
import pickle
import tarfile
import matplotlib.pyplot as plt
from tqdm import tqdm
import pygmtools
pygmtools.BACKEND = 'jittor'

class MyDataset(jt.dataset.Dataset):
    def __init__(self, data, labels):
        super().__init__()
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )

    def execute(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class DeepPermNet(nn.Module):
    def __init__(self):
        super(DeepPermNet, self).__init__()
        self.alexnet = AlexNet()
        self.fc7 = nn.Linear(512 * 4, 4096)
        self.fc8 = nn.Linear(4096, 16)

    def execute(self, x):
        x = self.alexnet(x)
        x = x.reshape(-1, 512*4)
        x = self.fc7(x) 
        x = self.fc8(x)
        x = x.reshape(-1, 4, 4)
        x = pygmtools.sinkhorn(x, max_iter=10)
        return jt.array(x)

# 显示数据集图片
def show_image(img1, img2, label=-1):
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck', 'picture']
    plt.subplot(1, 2, 1)
    plt.imshow(img1.transpose((1, 2, 0)).astype(int))
    plt.title(classes[label])
    plt.subplot(1, 2, 2)
    plt.imshow(img2.transpose((1, 2, 0)).astype(int))
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

# 打乱图片
def shuffle_image(image):
    # 分割图片
    upper_left = image[:, :16, :16]
    upper_right = image[:, :16, 16:]
    lower_left = image[:, 16:, :16]
    lower_right = image[:, 16:, 16:]
    patches = [upper_left, upper_right, lower_left, lower_right]
    
    # 原始顺序向量
    order = np.array([0, 1, 2, 3])
    order = order[np.newaxis, :]
    
    # 排列矩阵
    matrix = np.eye(4)
    np.random.shuffle(matrix)

    # 打乱顺序
    shuffled_order = order @ matrix
    shuffled_order = np.squeeze(shuffled_order).astype(int)
    
    upper_left_new = patches[shuffled_order[0] ]
    upper_right_new = patches[shuffled_order[1]]
    lower_left_new = patches[shuffled_order[2]]
    lower_right_new = patches[shuffled_order[3]]
    upper = np.concatenate((upper_left_new, upper_right_new), axis=2)
    lower = np.concatenate((lower_left_new, lower_right_new), axis=2)
    shuffled_image = np.concatenate((upper, lower), axis=1)
    shuffled_patches = np.array([upper_left_new, upper_right_new, lower_left_new, lower_right_new])

    # # 显示打乱前后的图片
    # show_image(image, shuffled_image)

    return shuffled_image, shuffled_patches, matrix

# 提取转移矩阵
def set_max_to_one(matrices):
    results = []
    for matrix in matrices:
        max_indices = np.argmax(matrix, axis=1)
        result = np.zeros_like(matrix)

        for col, row in enumerate(max_indices):
            result[col, row] = 1

        results.append(result)

    return np.array(results)

# 训练DPN
def train_dpn(model, train_loader, test_loader):

    criterion = nn.MSELoss()
    optimizer = nn.Adam(model.parameters(), lr=1e-4)

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    for epoch in range(50):
        print('Epoch: %d' % (epoch + 1))

        train_loss = 0
        correct = 0
        model.train()

        for i, (inputs, matrices) in tqdm(enumerate(train_loader), total=int(np.ceil(len(train_loader)/batch_size)), desc='Training'):
            inputs = inputs.view(-1, 3, 16, 16)
            outputs = model(inputs)
            loss = criterion(outputs, matrices)
            train_loss += np.squeeze(loss.numpy())
            optimizer.step(loss)

            predicted = set_max_to_one(outputs.data)
            mask = np.all(predicted == matrices.numpy(), axis=(1, 2))
            correct += np.count_nonzero(mask)


        print('Train Loss: %.3f Train Accuracy: %d %%' % (train_loss / int(np.ceil(len(train_loader)/batch_size)), 100 * correct / len(train_loader)))
        train_losses.append(train_loss / int(np.ceil(len(train_loader)/batch_size)))
        train_accs.append(100 * correct / len(train_loader))

        test_loss, test_acc = test_dpn(model, test_loader)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

    return model, train_losses, test_losses, train_accs, test_accs

# 测试DPN
def test_dpn(model, test_loader):
    model.eval()
    criterion = nn.MSELoss()
    correct = 0
    test_loss = 0

    with jt.no_grad():
        for i, (inputs, matrices) in tqdm(enumerate(test_loader), total=int(np.ceil(len(test_loader)/batch_size)), desc='Testing'):
            inputs = inputs.view(-1, 3, 16, 16)
            outputs = model(inputs)
            loss = criterion(outputs, matrices)
            test_loss += np.squeeze(loss.numpy())

            predicted = set_max_to_one(outputs.data)
            mask = np.all(predicted == matrices.numpy(), axis=(1, 2))
            correct += np.count_nonzero(mask)

    print('Test Loss: %.3f Test Accuracy: %d %%' % (test_loss / int(np.ceil(len(test_loader)/batch_size)), 100 * correct / len(test_loader)))

    return test_loss / int(np.ceil(len(test_loader)/batch_size)), 100 * correct / len(test_loader)


# 导入CIFAR-10数据集
tarfile_path = r"D:\Deep Learning and Its Application\Homework2\cifar-10-python.tar.gz"
train_data, train_labels, test_data, test_labels = load_data(tarfile_path)

train_data = jt.array(train_data).float32()
train_labels = jt.array(train_labels).int64()
test_data = jt.array(test_data).float32()
test_labels = jt.array(test_labels).int64()

shuffled_train_data = []
shuffled_test_data = []
train_matrix = []
test_matrix = []

# 打乱图片
for i in range(len(train_data)):
    shuffled_image, shuffled_patches, matrix = shuffle_image(train_data[i])
    shuffled_train_data.append(shuffled_patches)
    train_matrix.append(matrix)

shuffled_train_data = np.array(shuffled_train_data)
train_matrix = np.array(train_matrix)

for i in range(len(test_data)):
    shuffled_image, shuffled_patches, matrix = shuffle_image(test_data[i])
    shuffled_test_data.append(shuffled_patches)
    test_matrix.append(matrix)

shuffled_test_data = np.array(shuffled_test_data)
test_matrix = np.array(test_matrix)

# # 显示打乱前后的图片
# for i in range(len(train_data)):
#     image_parts = shuffled_train_data[i]
#     image_parts = image_parts.reshape(2, 2, 3, 16, 16)
#     image_parts = np.concatenate(image_parts, axis=3)
#     image = np.concatenate(image_parts, axis=1)
#     print(train_matrix[i])
#     show_image(train_data[i], image, train_labels[i])
    
# 创建数据加载器
train_dataset = MyDataset(shuffled_train_data, train_matrix)
test_dataset = MyDataset(shuffled_test_data, test_matrix)
batch_size = 256
train_loader = jt.dataset.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = jt.dataset.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 创建模型
model = DeepPermNet()

# 使用CUDA
jt.flags.use_cuda = 1

# 训练模型
model, train_losses, test_losses, train_accs, test_accs = train_dpn(model, train_loader, test_loader)

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
