import numpy as np  # TRY NOT TO USE THIS
import torch
import math
from collections import defaultdict

from torch import Tensor


class NeuralNetwork:
    def __init__(self, *argv):
        # return a dict of theta matrices
        # *argv will contain first the number of inputs, then any number of hidden layers, and then number of outputs
        self.theta = defaultdict(list)
        self.dE_dTheta = defaultdict(list)
        self.activations = defaultdict(list)
        self.weighted_sum = defaultdict(list)
        self.num_outputs = argv[-1]
        self.layer = argv[0:-1]
        print(self.layer)
        i = 0
        j = 0
        for arg in argv[0:-1]:
            self.theta[i] = (torch.randn(arg + 1, argv[i + 1])) / math.sqrt(arg)
            self.dE_dTheta[i] = torch.Tensor(arg + 1, argv[i + 1])
            i += 1
        for arg in argv[0:]:
            self.activations[j] = torch.ones(arg)
            self.weighted_sum[j] = torch.Tensor(arg)
            j = j + 1

    # print('dict activations shape', self.activations)

    # print('theta is',self.theta)
    # print('de by theta is',self.dE_dTheta)

    def getlayer(self, layer_num):
        return self.theta[layer_num]

    def forward(self, inp):

        # in_flat=inp.view(inp.numel()) #flatten a 2D Input
        # in_flat=[torch.Tensor.append(arg) for arg in argv]
        #  print('input size is ',inp.size())
        if len(inp.size()) == 1:
            inp_exp = torch.ones(inp.numel(), 1)
            inp_exp[:, 0] = inp
            self.activations[0] = inp_exp
            in_flat = torch.ones(inp.numel() + 1, 1)
            in_flat[1:, 0] = inp
            output = torch.Tensor()
        else:
            self.activations[0] = inp
            in_flat = torch.ones(inp.size(0) + 1, inp.size(1))
            in_flat[1:, :] = inp

            # print('input to nn.forward are:',in_flat)
            def sigmoid(arr):
                return 1 / (1 + torch.exp(-arr))

            # assume this dimension and
            #   self.activations[0]=in_flat
            self.weighted_sum[0] = 0
            for key, value in self.theta.items():
                if key == 0:
                    #  print('key0 position',in_flat,value)

                    z = torch.mm(torch.t(value), in_flat)  # for input
                    output = sigmoid(z)
                    self.weighted_sum[key + 1] = z
                    self.activations[key + 1] = output
                    #   if key==[final case]=output[key]=sigmoid(torch.cmul(output[key-1],value)) #dont apply sigmoid to output? you have to apply for now
                else:
                    #  print('loop1')
                    # output_mod=torch.ones(output.numel()+1,1)
                    output_mod = torch.ones(output.size(0) + 1, output.size(1))
                    output_mod[1:, :] = output
                    z = torch.mm(torch.t(value), output_mod)
                    output = sigmoid(z)
                    self.weighted_sum[key + 1] = z
                    self.activations[key + 1] = output
        output_batch = output
        return output_batch
        # print('dict activations shape', self.activations)
        # print('dict weighted sums shape', self.weighted_sum)

        return output_batch

    def backward(self, target, loss_criterion=None, eta=0.1, gd_bool=False):
        if loss_criterion == None: loss_criterion = 'MSE'
        numlayers = len(self.activations.keys())

        def diff_sigmoid(matrix):  # to get dOUT/dNET
            return matrix * (1 - matrix)  # only true for logistic, not for relu or other non-linearities

        def cost_prime(output, target):
            return output - target

        if len(target.size()) == 1:
            target_mid = torch.ones(target.size(0), 1)
            target_mid[:, 0] = target
            target = target_mid
        # print('target is',target)
        # print('output is',self.activations[numlayers-1])
        # print('problem vec is',len(self.theta.keys()))
        act = self.activations[numlayers - 1]
        diff = target - act
        MSE_loss = torch.sum(torch.pow(diff, 2), 0) * 0.5
        # print('MSE Loss is ', MSE_loss)
        CE_Loss = torch.sum((-target * torch.log(act) - (1 - target) * torch.log(1 - act)))
        # print('CE Loss is', CE_Loss)
        if loss_criterion == 'MSE':
            delta = cost_prime(act, target) * diff_sigmoid(act)
        elif loss_criterion == 'CE':
            delta = cost_prime(act, target)

        new_array = torch.Tensor(self.dE_dTheta[numlayers - 2].size())  # type: Tensor
        new_array[0, :] = delta.mean(1)  # for bias
        new_array[1:, :] = torch.mm(self.activations[numlayers - 2], torch.t(delta)) / delta.size(1)  # for weights
        self.dE_dTheta[numlayers - 2] = new_array

        # print('dE_dTheta for last layer is', self.dE_dTheta[numlayers-2])

        # backpropagate this into hidden layers:
        numHiddenLayers = numlayers - 2
        last_hidden_idx = len(self.layer) - 1
        if gd_bool:
            self.theta[last_hidden_idx] = self.theta[last_hidden_idx] - eta * self.dE_dTheta[last_hidden_idx]

        for hid in range(numHiddenLayers):
            # print('in hidden layer from back and from front =',hid,-hid-1)
            relevant_weights = self.theta[numlayers - 2 - hid][1:, :]
            relevant_activations = self.activations[numlayers - hid - 2]
            relevant_activations_prev = self.activations[numlayers - hid - 3]

            # print('relevant_weights are',relevant_weights)
            # print('relevant_activations are',relevant_activations)
            # print('problem vec is ',relevant_weights.size(),delta.size(),relevant_activations.size())
            # print('check sizes,',relevant_weights,delta,relevant_activations)
            if loss_criterion == 'MSE':
                delta = torch.mm(relevant_weights, delta) * diff_sigmoid(relevant_activations)
            elif loss_criterion == 'CE':
                delta = torch.mm(relevant_weights, delta) * diff_sigmoid(relevant_activations)
            # print('delta for this hidden layer is or dE/dneth1', delta)  #this corresponds to dE/dneth1

            self.dE_dTheta[numlayers - 3 - hid][0, :] = delta.mean(1)  # for bias
            # print('problem vec is relevant_activations_prev,torch.t(delta)) ',relevant_activations_prev,torch.t(delta))
            self.dE_dTheta[numlayers - 3 - hid][1:, :] = torch.mm(relevant_activations_prev,
                                                                  torch.t(delta)) / delta.size(1)  # for weights
            # print('dE_dTheta for this layer is', self.dE_dTheta[numlayers-3-hid])
            if gd_bool:
                self.theta[numlayers - 3 - hid] = self.theta[numlayers - 3 - hid] - eta * self.dE_dTheta[
                    numlayers - 3 - hid]

        # self.theta[0] = self.theta[0] - 1 * self.dE_dTheta[0]
        # print('final delta is',self.dE_dTheta)
        if loss_criterion == 'MSE':
            return MSE_loss  # TODO: remove later, just for testing
        elif loss_criterion == 'CE':
            return CE_Loss

    # # this could not be bypassed, the list limitation just like numpy
    # def updateParams(self, eta):
    #     # print('original theta is ', self.theta)
    #     for key, value in self.theta.items():
    #         #  if key==0:
    #         #  print('key0 position',in_flat,value)
    #         self.theta[key] = self.theta[key] - eta * self.dE_dTheta[key]
    #         # print('thetakey0',self.theta[key])
    #         # print('dthetakey0',self.dE_dTheta[key])
    #         # print('updated Theta is', self.theta)
