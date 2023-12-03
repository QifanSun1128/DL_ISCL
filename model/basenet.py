from torchvision.models import alexnet, vgg16, alexnet_weights, vgg16_weights
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Function


class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output * -ctx.lambd), None


def grad_reverse(x, lambd=1.0):
    model = GradReverse.apply(x, lambd)
    return model


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)

    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)

    _output = torch.div(input, norm.view(-1, 1).expand_as(input))

    output = _output.view(input_size)

    return output


class AlexNetBase(nn.Module):
    def __init__(self, pret=True):
        super(AlexNetBase, self).__init__()

        # Load the AlexNet model with or without pretrained weights
        if pret:
            model_alexnet = alexnet(weights=alexnet_weights.ALEXNET_IMAGENET1K_V1)
        else:
            model_alexnet = alexnet(weights=None)

        # Use the features from AlexNet
        self.features = nn.Sequential(*model_alexnet.features)

        # Reconstruct the classifier, omitting the last layer
        self.classifier = nn.Sequential(*model_alexnet.classifier[:-1])

        # Store the number of input features for the last layer
        self.__in_features = model_alexnet.classifier[-1].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def output_num(self):
        return self.__in_features


class VGGBase(nn.Module):
    def __init__(self, pret=True, no_pool=False):
        super(VGGBase, self).__init__()

        if pret:
            vgg16_model = vgg16(weights=vgg16_weights.VGG16_IMAGENET1K_V1)
        else:
            vgg16_model = vgg16(weights=None)

        # If no pooling is required, remove the last pooling layer from features
        if no_pool:
            features = list(vgg16_model.features.children())[:-1]
        else:
            features = list(vgg16_model.features.children())

        self.features = nn.Sequential(*features)

        classifier = list(vgg16_model.classifier.children())[:-1]
        self.classifier = nn.Sequential(*classifier)

        self.s = nn.Parameter(torch.FloatTensor([10]))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 7 * 7 * 512)
        x = self.classifier(x)
        return x


class Predictor(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(Predictor, self).__init__()
        self.fc = nn.Linear(inc, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=0.1):
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc(x) / self.temp
        return x_out


class Predictor_deep(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(Predictor_deep, self).__init__()
        self.fc1 = nn.Linear(inc, 512)
        self.fc2 = nn.Linear(512, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=0.1):
        x = self.fc1(x)
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc2(x) / self.temp
        return x_out


class Discriminator(nn.Module):
    def __init__(self, inc=4096):
        super(Discriminator, self).__init__()
        self.fc1_1 = nn.Linear(inc, 512)
        self.fc2_1 = nn.Linear(512, 512)
        self.fc3_1 = nn.Linear(512, 1)

    def forward(self, x, reverse=True, eta=1.0):
        if reverse:
            x = grad_reverse(x, eta)
        x = F.relu(self.fc1_1(x))
        x = F.relu(self.fc2_1(x))
        x_out = torch.sigmoid(self.fc3_1(x))
        return x_out
