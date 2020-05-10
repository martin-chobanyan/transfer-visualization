import os

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm

from dataset import ImagePairs
from train_utils import create_folder, get_device, Logger
from feature_vis.transforms import ImagenetNorm

# define the constants
ROOT_DIR = '/home/mchobanyan/data/research/transfer/vis/'
IMAGENET_DIR = os.path.join(ROOT_DIR, 'pretrained-resnet50')
DOG_DIR = os.path.join(ROOT_DIR, 'finetune-dog-resnet50')
CAR_DIR = os.path.join(ROOT_DIR, 'finetune-car-resnet50')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'comparisons')
LAYERS = ['layer3-bottleneck5', 'layer4-bottleneck0', 'layer4-bottleneck1', 'layer4-bottleneck2']
DEVICE = get_device()
BATCH_SIZE = 100


def compare_image_dirs(image_pairs, distance_fns, output_path, device, batch_size=BATCH_SIZE):
    logger = Logger(output_path, header=['channel'] + list(distance_fns.keys()))
    loader = DataLoader(image_pairs, batch_size=batch_size, shuffle=False)
    for imgs1, imgs2 in tqdm(loader):
        imgs1  = imgs1.to(device)
        imgs2  = imgs2.to(device)
        for fn_name, dist_fn in distance_fns.items():
            d = dist_fn(imgs1, imgs2)
            print(fn_name, d)
        break


def main():
    create_folder(OUTPUT_DIR)
    image_transforms = Compose([ToTensor(), ImagenetNorm()])
    distance_fns = {
        'gram_layer1': None,
        'gram_layer2': None,
        'gram_layer3': None,
        'gram_layer4': None,
        'gram_total': None
    }

    # compare resnet50 pre-trained on ImageNet vs fine-tuned on the dog breed dataset
    for layer in LAYERS:
        img_dir1 = os.path.join(IMAGENET_DIR, 'features', layer)
        img_dir2 = os.path.join(DOG_DIR, 'features', layer)
        image_pairs = ImagePairs(img_dir1, img_dir2, image_transforms)

        output_path = os.path.join(OUTPUT_DIR, f'imagenet-vs-dogs-{layer}.csv')

        compare_image_dirs(image_pairs, distance_fns, output_path, DEVICE)
        break


if __name__ == '__main__':
    main()
