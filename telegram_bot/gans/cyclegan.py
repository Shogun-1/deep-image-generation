import torch.utils.data.distributed
import torchvision.transforms as transforms

from PIL import Image

from gans.models import Generator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMSIZE = 256

CYCLEGAN_TASK_LIST = ['horse2zebra', 'zebra2horse', 'apple2orange', 'orange2apple', 'cezanne2photo', 'photo2cezanne']

preprocess = transforms.Compose([transforms.Resize(IMSIZE),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                 ])


def image_loader(image_name):
    image = Image.open(image_name)
    image = preprocess(image).unsqueeze(0)
    return image.to(DEVICE, torch.float)


toImage = transforms.ToPILImage()


def prepare_image(tensor):
    tensor = tensor.cpu().clone()
    tensor = tensor.squeeze(0)
    image = toImage(tensor)
    return image


class CycleGAN:
    def __init__(self):
        self.model = Generator().to(DEVICE)

    def set_task(self, task='horse2zebra'):
        weights = f'gans/weights/cyclegan/netG/{task}.pth'

        self.model.load_state_dict(torch.load(weights, map_location=DEVICE))
        self.model.eval()

    def generate(self, img_link):
        image = image_loader(img_link)
        fake = self.model(image).detach().cpu()
        fake = (fake + 1) / 2
        fake = prepare_image(fake)
        return fake
