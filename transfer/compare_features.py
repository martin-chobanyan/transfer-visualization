import os
from statistics import mean

from torch import no_grad
from torchvision.models import resnet50
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm

from dataset import ImagePairs
from distance import GramDistanceResnet50
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


def compare_images(image_pairs, gram_fn, output_path, device, transforms, title=''):
    logger = Logger(output_path, header=['channel', 'gram_avg'] + [f'gram_layer{i}' for i in range(1, 5)])
    for channel_idx in tqdm(range(len(image_pairs)), desc=title):
        img1, img2 = image_pairs[channel_idx]
        img1 = transforms(img1).unsqueeze(0).to(device)
        img2 = transforms(img2).unsqueeze(0).to(device)

        # calculate the gram distances
        with no_grad():
            gram_distances = gram_fn(img1, img2).squeeze().tolist()
        gram_avg = mean(gram_distances)

        logger.add_entry(*([channel_idx, gram_avg] + gram_distances))


def run(imagenet_dir, target_dir, output_dir):
    create_folder(output_dir)
    image_transforms = Compose([ToTensor(), ImagenetNorm()])

    model = resnet50(pretrained=True)
    model = model.to(DEVICE)
    model = model.eval()
    gram_distance = GramDistanceResnet50(model)

    # compare resnet50 pre-trained on ImageNet vs fine-tuned on the dog breed dataset
    for layer in LAYERS:
        img_dir1 = os.path.join(imagenet_dir, 'features', layer)
        img_dir2 = os.path.join(target_dir, 'features', layer)
        image_pairs = ImagePairs(img_dir1, img_dir2)

        output_path = os.path.join(output_dir, f'imagenet-vs-dogs-{layer}.csv')
        compare_images(image_pairs, gram_distance, output_path, DEVICE, image_transforms, layer)


if __name__ == '__main__':
    print('Comparing ImageNet vs Dog Breeds feature visualizations')
    # run(IMAGENET_DIR, DOG_DIR, os.path.join(OUTPUT_DIR, 'dogs'))
    print()

    print('Comparing ImageNet vs Car Models feature visualizations')
    run(IMAGENET_DIR, CAR_DIR, os.path.join(OUTPUT_DIR, 'cars'))
