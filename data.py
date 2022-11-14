import os
from PIL import Image, ImageEnhance
import random
import numpy as np
import cv2

import torch
import torch.utils.data as data
import torchvision.transforms as transforms


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

#several data augumentation strategies
def cv_random_flip(img, label, depth):
    flip_flag = random.randint(0, 1)
    #left right flip
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label, depth

def randomCrop(image, label, depth):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width-border , image_width)
    crop_win_height = np.random.randint(image_height-border , image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region),depth.crop(random_region)

def randomRotation(image, label, depth):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        depth = depth.rotate(random_angle, mode)
    return image, label, depth

def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image

def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im
    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))

def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)
        if random.randint(0, 1) == 0:
            img[randX,randY] = 0
        else:
            img[randX,randY] = 255
    return Image.fromarray(img)


class SalObjDataset(data.Dataset):
    def __init__(
            self, image_root, depth_root, gt_root, mask_root, gray_root, trainsize, warmup_stage=True
    ):
        self.trainsize = trainsize
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.depths = [os.path.join(depth_root, f) for f in os.listdir(depth_root) if f.endswith('.png')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.masks = [os.path.join(mask_root, f) for f in os.listdir(mask_root) if f.endswith('.png')]
        self.grays = [os.path.join(gray_root, f) for f in os.listdir(gray_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.depths = sorted(self.depths)
        self.gts = sorted(self.gts)
        self.masks = sorted(self.masks)
        self.grays = sorted(self.grays)
        self.filter_files()
        self.size = len(self.images)
        self.resize_transform = transforms.Resize((self.trainsize, self.trainsize))
        self.to_tensor_transform = transforms.ToTensor()
        self.normalize_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.warmup_stage = warmup_stage

    def __getitem__(self, index):
        # prepare inputs for warmup model or student model
        image = self.rgb_loader(self.images[index])
        depth = self.rgb_loader(self.depths[index])
        gt = self.binary_loader(self.gts[index])
        mask = self.binary_loader(self.masks[index])
        gray = self.binary_loader(self.grays[index])

        if not self.warmup_stage:
            image = colorEnhance(image)
        image = self.resize_transform(image)
        image_edge = self.to_tensor_transform(self.canny_edge_generator(image))
        image = self.normalize_transform(self.to_tensor_transform(image))
        depth = self.resize_transform(depth)
        depth_edge = self.to_tensor_transform(self.canny_edge_generator(depth))
        depth = self.to_tensor_transform(depth)
        gt = self.to_tensor_transform(self.resize_transform(gt))
        mask = self.to_tensor_transform(self.resize_transform(mask))
        gray = self.to_tensor_transform(self.resize_transform(gray))

        # prepare inputs for teacher model
        if not self.warmup_stage:
            image_teacher = self.rgb_loader(self.images[index])
            depth_teacher = self.rgb_loader(self.depths[index])

            flip_flag = random.randint(0, 1)
            if flip_flag == 1:
                image_teacher = image_teacher.transpose(Image.FLIP_LEFT_RIGHT)
                depth_teacher = depth_teacher.transpose(Image.FLIP_LEFT_RIGHT)
            image_edge_teacher = self.to_tensor_transform(
                self.canny_edge_generator(self.resize_transform(image_teacher))
            )
            image_teacher = self.normalize_transform(self.to_tensor_transform(self.resize_transform(image_teacher)))
            depth_edge_teacher = self.to_tensor_transform(
                self.canny_edge_generator(self.resize_transform(depth_teacher))
            )
            depth_teacher = self.to_tensor_transform(self.resize_transform(depth_teacher))

            return image, depth, gt, mask, gray, image_edge, depth_edge, \
                   image_teacher, depth_teacher, image_edge_teacher, depth_edge_teacher, flip_flag
        else:
            return image, depth, gt, mask, gray, image_edge, depth_edge

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        assert len(self.images) == len(self.depths)
        images = []
        depths = []
        gts = []
        masks = []
        grays = []
        for img_path, depth_path, gt_path, mask_path, gray_path in zip(
                self.images, self.depths, self.gts, self.masks, self.grays
        ):
            img = Image.open(img_path)
            depth = Image.open(depth_path)
            gt = Image.open(gt_path)
            mask = Image.open(mask_path)
            gray = Image.open(gray_path)
            if img.size == gt.size:
                images.append(img_path)
                depths.append(depth_path)
                gts.append(gt_path)
                masks.append(mask_path)
                grays.append(gray_path)
        self.images = images
        self.depths = depths
        self.gts = gts
        self.masks = masks
        self.grays = grays

    def canny_edge_generator(self, image: Image.Image) -> np.array:
        edge = cv2.Canny(np.array(image).astype(np.uint8), 10, 100)
        return edge

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


def get_loader(
        image_root, depth_root, gt_root, mask_root, gray_root, batchsize, trainsize,
        shuffle=True, num_workers=12, pin_memory=True, warmup_stage=True
):
    dataset = SalObjDataset(
        image_root, depth_root, gt_root, mask_root, gray_root, trainsize, warmup_stage
    )
    data_loader = data.DataLoader(
        dataset=dataset, batch_size=batchsize, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory
    )
    return data_loader


class test_dataset:
    def __init__(self, image_root, depth_root, testsize):
        self.testsize = testsize
        self.images = [
            os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')
        ]
        self.depths = [
            os.path.join(depth_root, f) for f in os.listdir(depth_root) if f.endswith('.bmp') or f.endswith('.png')
        ]
        self.images = sorted(self.images)
        self.depths = sorted(self.depths)

        self.resize_transform = transforms.Resize((self.testsize, self.testsize))
        self.to_tensor_transform = transforms.ToTensor()
        self.normalize_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        depth = self.rgb_loader(self.depths[self.index])
        ori_height, ori_width = image.size[0], image.size[1]

        image_edge = self.to_tensor_transform(self.canny_edge_generator(self.resize_transform(image))).unsqueeze(0)
        image = self.resize_transform(image)
        image = self.normalize_transform(self.to_tensor_transform(image)).unsqueeze(0)

        depth_edge = self.to_tensor_transform(self.canny_edge_generator(self.resize_transform(depth))).unsqueeze(0)
        depth = self.resize_transform(depth)
        depth = self.to_tensor_transform(depth).unsqueeze(0)

        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, depth, ori_height, ori_width, name, image_edge, depth_edge

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def canny_edge_generator(self, image: Image.Image) -> np.array:
        edge = cv2.Canny(np.array(image).astype(np.uint8), 10, 100)
        return edge