import torch
import math
from collections import defaultdict
from torchvision.datasets import MNIST
from torchvision import transforms
from neural_network import NeuralNetwork
import torch.utils.data.dataloader as dataloader
import time
import numpy as np


class MyImg2Num:
    def __init__(self):
        self.train_batch_size = 64
        self.test_batch_size = 1000
        self.input_size = 784
        self.output_size = 10
        self.model = NeuralNetwork(784, 500, 100, 10)
        self.image_h = 28
        self.image_w = 28
        self.image_d = 1
        self.pixels = self.image_h * self.image_w * self.image_d
        self.learning_rate = 0.1
        self.num_epochs = 4

    def forward(self, img_batch):
        input_vector = img_batch.view(-1, self.pixels)
        output = self.model.forward(torch.t(input_vector))
        return torch.t(output)

    def train(self):
        def one_hot_encoding(input_num):
            one_hot = torch.zeros(10, dtype=torch.short)
            one_hot[input_num] = 1
            return one_hot

        loss_train = []
        loss_test = []
        correct_batch = 0
        correct_batch_test = 0
        correct_epoch = []
        correct_epoch_test = []

        ###LOAD DATA
        train = MNIST('./data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,)),# ToTensor does min-max normalization.
        ]), )

        test = MNIST('./data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,)),# ToTensor does min-max normalization.
        ]), )

        # Create DataLoader
        dataloader_args = dict(shuffle=True, batch_size=64)  # data loader reshuffles every step
        train_loader = dataloader.DataLoader(train, **dataloader_args)
        test_loader = dataloader.DataLoader(test, **dataloader_args)

        train_data = train.train_data
        train_data = train.transform(train_data.numpy())

        ###TRAINING
        time_meas = torch.Tensor(self.num_epochs)
        acc_train = torch.Tensor(self.num_epochs)
        error_train = torch.Tensor(self.num_epochs)
        start_time = time.clock()
        for epoch in range(self.num_epochs):
            print('----------------- the epoch {:d} -----------------'.format(epoch))
            if (epoch == 1):
                print("epoch = 1")
            print('training epoch ', epoch)
            for batch_idx, (data, target) in enumerate(train_loader):
                batch_size = data.size(0)
                input_64x784 = data.view(batch_size, self.pixels)
                target_64x10 = torch.Tensor(batch_size, self.output_size)
                for i in range(batch_size): target_64x10[i, :] = one_hot_encoding(target[i])
                out = self.forward(input_64x784)
                correct_batch = correct_batch + torch.sum(out.max(1)[1] == target)
                # print('out max(1) is and target is', out.max(1), target)
                # print(correct_batch)

                # where all bp happens here
                loss = (
                    self.model.backward(torch.t(target_64x10), gd_bool=True).mean())  ## it replaces the updateParams
                loss_train.append(loss)
                # self.model.updateParams()

                # print('debug')
            correct_epoch.append(correct_batch)
            print('train accuracy is {:.1f}% and loss is {:f}'.format(correct_epoch[-1].to(dtype=torch.float) / 600,
                                                                      loss_train[-1]))
            print('seconds elapsed is ', time.clock() - start_time)
            time_meas[epoch] = time.clock() - start_time
            acc_train[epoch] = correct_epoch[-1].to(dtype=torch.float) / 600
            error_train[epoch] = loss_train[-1]
            for batch_idx_test, (data_test, target_test) in enumerate(test_loader):
                batch_size_test = data_test.size(0)

                # print(batch_size_test)
                input_64x784 = data_test.view(batch_size_test, self.pixels)
                target_64x10 = torch.Tensor(batch_size_test, self.output_size)
                # print(target_64x10.shape)l
                for i in range(batch_size_test): target_64x10[i, :] = one_hot_encoding(target_test[i])
                out_test = self.forward(input_64x784)
                # print(out_test.shape)
                correct_batch_test = correct_batch_test + torch.sum(out_test.max(1)[1] == target_test)
                # print(correct_batch_test)
                loss_test.append(self.model.backward(torch.t(target_64x10)).mean())
            correct_epoch_test.append(correct_batch_test)
            print('For the epoch {:d}, test accuracy is {:.1f}% and loss is {:f}'.format(epoch,
                                                                                         correct_epoch_test[-1].to(
                                                                                             dtype=torch.float) / 100,
                                                                                         loss_test[-1]))
            correct_batch = 0  # for each epoch
            correct_batch_test = 0
        correct_epoch_percent = [correct_epoch[i] / 600 for i in range(len(correct_epoch))]
        print('Here is your time to train', time_meas)
        print('Here is your train error', error_train)
        print('Here is your accuracy', acc_train)

        ### TESTING
        for batch_idx_test, (data_test, target_test) in enumerate(test_loader):
            batch_size_test = data_test.size(0)

            # print(batch_size_test)
            input_64x784 = data_test.view(batch_size_test, self.pixels)
            target_64x10 = torch.Tensor(batch_size_test, self.output_size)
            # print(target_64x10.shape)
            for i in range(batch_size_test): target_64x10[i, :] = one_hot_encoding(target_test[i])
            out_test = self.forward(input_64x784)
            # print(out_test.shape)
            correct_batch_test = correct_batch_test + torch.sum(out_test.max(1)[1] == target_test)
            # print(correct_batch_test)
            loss_test.append(self.model.backward(torch.t(target_64x10)).mean())
        correct_epoch_test.append(correct_batch_test)
        print('test accuracy is {:.1f}% and loss is {:f}'.format(correct_epoch_test[-1].to(dtype=torch.float) / 100,
                                                                 loss_test[-1]))
        print('end')

# if __name__ == "__main__":
#     main()
