import torch
import torchvision.transforms as transforms
from PIL import Image

from gans.models import GeneratorPix2Pix

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMSIZE = 256

PIX2PIX_TASK_LIST = ['day2night']

transformations = transforms.Compose([
    transforms.Resize(IMSIZE),
    transforms.CenterCrop(IMSIZE),
    transforms.ToTensor()])


def image_loader(image_name):
    image = Image.open(image_name)
    image = transformations(image).unsqueeze(0).to(DEVICE, torch.float)
    # image = image / 127.5 - 1
    return image


toImage = transforms.ToPILImage()


def prepare_image(tensor):
    tensor = tensor.cpu().clone()
    # tensor = (tensor + 1) * 127.5
    tensor = tensor.squeeze(0)
    image = toImage(tensor)
    return image


class Pix2Pix:
    def __init__(self):
        self.model = GeneratorPix2Pix().to(DEVICE)
        self.model.load_state_dict(torch.load('gans/weights/pix2pix/day2night.pth', map_location=DEVICE))
        self.model.eval()

    def generate(self, img_link):
        image = image_loader(img_link)
        fake = self.model(image).detach().cpu()
        processed_fake = prepare_image(fake)
        return processed_fake
