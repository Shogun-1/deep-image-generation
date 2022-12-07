import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMSIZE = 256

preprocess = transforms.Compose([transforms.Resize(IMSIZE),
                                 transforms.CenterCrop(IMSIZE),
                                 transforms.ToTensor()])


def image_loader(image, url=True):
    if url:
        image = Image.open(image)
    image = preprocess(image).unsqueeze(0)
    return image.to(DEVICE, torch.float)


toImage = transforms.ToPILImage()


def prepare_image(tensor):
    tensor = tensor.cpu().clone()
    tensor = tensor.squeeze(0)
    image = toImage(tensor)
    return image


def get_mask_3d(input_, style_location='all'):
    # height == width always in this architecture
    batch_size, channels, height, width = input_.size()

    # only one style for picture used
    if style_location == 'all':
        mask_img_2d = torch.ones(height)

    # style for the lower left part of the picture
    elif style_location == 'lower_left':
        mask_img_2d = torch.eye(height)
        for h in range(height):
            for w in range(width):
                if mask_img_2d[h, w] == 1:
                    mask_img_2d[h, :w] = 1
                    break

    # style for the upper right part of the picture
    elif style_location == 'upper_right':
        mask_img_2d = torch.eye(height)
        for h in range(height):
            for w in range(width):
                if mask_img_2d[h, w] == 1:
                    mask_img_2d[h, w:] = 1
                    break
    else:
        raise AttributeError("style_number should be 0, 1 or 2")

    mask_img_3d = torch.empty_like(input_, requires_grad=False)
    mask_img_3d[0, :] = mask_img_2d

    return mask_img_3d


class ContentLoss(nn.Module):

    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = F.mse_loss(self.target, self.target)

    def forward(self, input_):
        self.loss = F.mse_loss(input_, self.target)
        return input_


def gram_matrix(input_):
    batch_size, c, h, w = input_.size()

    features = input_.view(batch_size * c, -1)

    g = torch.mm(features, features.t())

    return g.div(batch_size * c * h * w)


class StyleLoss(nn.Module):
    def __init__(self, target_feature, mask_image):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.mask = mask_image.detach()
        self.loss = F.mse_loss(self.target, self.target)

    def forward(self, input_):
        b, c, h, w = input_.size()

        mask_img_2d = F.interpolate(self.mask, size=(h, w))[0, 0]
        mask_img_3d = torch.empty_like(input_, requires_grad=False)
        mask_img_3d[0, :] = mask_img_2d

        input_ = input_ * mask_img_3d

        g = gram_matrix(input_)
        self.loss = F.mse_loss(g, self.target)
        return input_


cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(DEVICE)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(DEVICE)


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

convnet = models.vgg19(pretrained=True).features.to(DEVICE).eval()


class NSTModel:
    def __init__(self, content_img, style_img, style_location='all', cnn=convnet,
                 normalization_mean=cnn_normalization_mean, normalization_std=cnn_normalization_std,
                 content_layers=None, style_layers=None):
        if style_layers is None:
            style_layers = style_layers_default
        if content_layers is None:
            content_layers = content_layers_default

        if not isinstance(content_img, torch.Tensor):
            if isinstance(content_img, Image.Image):
                content_img = image_loader(content_img, url=False)
            else:
                content_img = image_loader(content_img, url=True)

        self.content_img = content_img
        self.style_img = image_loader(style_img)
        self.out_img = copy.deepcopy(content_img)
        self.mask_img = get_mask_3d(content_img, style_location=style_location)

        cnn = copy.deepcopy(cnn)

        normalization = Normalization(normalization_mean, normalization_std).to(DEVICE)

        content_losses = []
        style_losses = []

        model = nn.Sequential(normalization)

        i = 0
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in content_layers:
                target = model(self.content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                target_feature = model(self.style_img).detach()
                style_loss = StyleLoss(target_feature, self.mask_img)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        # Now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        self.model = model[:(i + 1)]
        self.content_losses = content_losses
        self.style_losses = style_losses

    def fit(self, num_steps=400, style_weight=100000, content_weight=1):
        optimizer = optim.LBFGS([self.out_img.requires_grad_()])

        run = [0]
        while run[0] <= num_steps:

            def closure():
                self.out_img.data.clamp_(0, 1)

                optimizer.zero_grad()

                self.model(self.out_img)

                style_score = 0
                content_score = 0

                for sl in self.style_losses:
                    style_score += sl.loss
                for cl in self.content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1

                return style_score + content_score

            optimizer.step(closure)

        self.out_img.data.clamp_(0, 1)

    def get_image(self):
        output = prepare_image(self.out_img)
        return output
