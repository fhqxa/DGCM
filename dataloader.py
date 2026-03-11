
import os
import os.path as osp
import numpy as np
import torch
import csv
import json
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import pickle
from collections import defaultdict
from PIL import Image


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
BICUBIC = InterpolationMode.BICUBIC

clip_preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224, interpolation=BICUBIC),
    transforms.CenterCrop(224),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(CLIP_MEAN, CLIP_STD),
])

clip_preprocess_from_pil = transforms.Compose([
    transforms.Resize(224, interpolation=BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(CLIP_MEAN, CLIP_STD),
])

def wnid_to_name(wnid):

    try:
        from nltk.corpus import wordnet as wn

        if not wnid.startswith('n') or len(wnid) != 9:
            return wnid

        offset = int(wnid[1:])
        synset = wn.synset_from_pos_and_offset('n', offset)

        lemma_names = synset.lemma_names()

        if not lemma_names:
            return wnid


        multi_word_names = [n for n in lemma_names if '_' in n]
        single_word_names = [n for n in lemma_names if '_' not in n]

        if multi_word_names:
            best_name = min(multi_word_names, key=lambda x: len(x))
        else:
            best_name = min(single_word_names, key=lambda x: len(x))

        best_name = best_name.replace('_', ' ')

        return best_name

    except ImportError:
        print("Warning: nltk not installed. Install via 'pip install nltk'")
        print("Then run: python -m nltk.downloader wordnet")
        return wnid
    except Exception as e:
        return wnid


class MiniImageNet(Dataset):


    def __init__(self, mode, root, num_ways, num_shots, **kwargs):
        self.root = osp.join(root, 'mini-imagenet')
        self.mode = mode
        self.num_way = num_ways
        self.num_shot = num_shots
        self.num_query = kwargs.get('num_query', 15)

        csv_file = osp.join(self.root, 'split', f'{self.mode}.csv')
        if not osp.exists(csv_file):
            csv_file = osp.join(self.root, f'{self.mode}.csv')
            if not osp.exists(csv_file):
                raise FileNotFoundError(f"Split CSV not found at: {csv_file}")

        print(f"Loading {self.mode} data from: {csv_file}")

        self.class_dict = defaultdict(list)
        image_root = osp.join(self.root, 'images')

        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            try:
                next(reader)
            except StopIteration:
                raise ValueError(f"CSV file {csv_file} is empty!")

            for row in reader:
                if len(row) < 2: continue
                filename, label_name = row[0], row[1]
                img_path = osp.join(image_root, filename)
                if not osp.exists(img_path):
                    img_path = osp.join(image_root, label_name, filename)
                if osp.exists(img_path):
                    self.class_dict[label_name].append(img_path)

        self.classes = list(self.class_dict.keys())
        self.num_classes = len(self.classes)

        if self.num_classes == 0:
            raise RuntimeError(f"No valid classes found in {csv_file}")

        print(f"Converting WordNet IDs to readable names...")
        self.label_name = {}
        conversion_count = 0

        for class_key in self.classes:
            readable_name = wnid_to_name(class_key)
            self.label_name[class_key] = readable_name

            if readable_name != class_key:
                conversion_count += 1

            if len(self.label_name) <= 5:
                print(f"  {class_key} → {readable_name}")

        print(f"Successfully converted {conversion_count}/{self.num_classes} class names")

        self.transform = clip_preprocess_from_pil
        print(f"INFO: [MiniImageNet] {self.mode} - {self.num_classes} classes, "
              f"{sum(len(v) for v in self.class_dict.values())} images")

    def __len__(self):
        return 10000

    def __getitem__(self, index):
        selected_class_keys = np.random.choice(self.classes, self.num_way, replace=False)
        support_x, support_y, query_x, query_y = [], [], [], []
        episode_class_names = [self.label_name[key] for key in selected_class_keys]

        for i, class_key in enumerate(selected_class_keys):
            image_paths = self.class_dict[class_key]
            required_imgs = self.num_shot + self.num_query
            replace_flag = len(image_paths) < required_imgs
            selected_paths = np.random.choice(image_paths, required_imgs, replace=replace_flag)

            for path in selected_paths[:self.num_shot]:
                image = Image.open(path).convert("RGB")
                support_x.append(self.transform(image))
                support_y.append(i)

            for path in selected_paths[self.num_shot:]:
                image = Image.open(path).convert("RGB")
                query_x.append(self.transform(image))
                query_y.append(i)

        return (torch.stack(support_x), torch.LongTensor(support_y),
                torch.stack(query_x), torch.LongTensor(query_y),
                episode_class_names)


class TieredImageNet(Dataset):


    def __init__(self, mode, root, num_ways, num_shots, **kwargs):
        self.root = osp.join(root, 'tiered_imagenet')
        self.mode = mode
        self.num_way = num_ways
        self.num_shot = num_shots
        self.num_query = kwargs.get('num_query', 15)

        # 读取类名映射文件
        class_names_json = osp.join(self.root, 'class_names.json')
        if not osp.exists(class_names_json):
            raise FileNotFoundError(f"class_names.json not found at: {class_names_json}")

        print(f"Loading class names from: {class_names_json}")
        with open(class_names_json, 'r') as f:
            self.wnid_to_name = json.load(f)

        print(f"Loaded {len(self.wnid_to_name)} WordNet ID mappings")

        # 数据目录
        mode_dir = osp.join(self.root, mode)
        if not osp.exists(mode_dir):
            raise FileNotFoundError(f"{mode} directory not found at: {mode_dir}")

        print(f"Loading images from: {mode_dir}")


        self.class_dict = defaultdict(list)
        self.wnid_to_readable = {}


        class_folders = [d for d in os.listdir(mode_dir)
                         if osp.isdir(osp.join(mode_dir, d))]

        for class_folder in sorted(class_folders):

            wnid = class_folder
            class_dir = osp.join(mode_dir, class_folder)


            if wnid in self.wnid_to_name:
                readable_name = self.wnid_to_name[wnid]
                if ',' in readable_name:
                    readable_name = readable_name.split(',')[0].strip()
                self.wnid_to_readable[wnid] = readable_name
            else:
                self.wnid_to_readable[wnid] = wnid
                print(f"Warning: WordNet ID {wnid} not found in class_names.json")

            image_files = [f for f in os.listdir(class_dir)
                           if f.lower().endswith(('.jpeg', '.jpg', '.png', '.bmp'))]

            for img_file in image_files:
                img_path = osp.join(class_dir, img_file)
                self.class_dict[wnid].append(img_path)

        self.classes = sorted(self.class_dict.keys())
        self.num_classes = len(self.classes)

        if self.num_classes == 0:
            raise RuntimeError(f"No classes found in {mode_dir}")

        self.transform = clip_preprocess_from_pil

        total_images = sum(len(v) for v in self.class_dict.values())
        print(f"INFO: [TieredImageNetFolder] {mode} - {self.num_classes} classes, "
              f"{total_images} images")

        print("Sample classes:")
        for wnid in self.classes[:3]:
            readable = self.wnid_to_readable[wnid]
            count = len(self.class_dict[wnid])
            print(f"  {wnid} → '{readable}' ({count} images)")

    def __len__(self):
        return 10000

    def __getitem__(self, index):
        selected_wnids = np.random.choice(self.classes, self.num_way, replace=False)
        support_x, support_y, query_x, query_y = [], [], [], []

        episode_class_names = [self.wnid_to_readable[wnid] for wnid in selected_wnids]

        for i, wnid in enumerate(selected_wnids):
            paths = self.class_dict[wnid]
            required_imgs = self.num_shot + self.num_query

            replace_flag = len(paths) < required_imgs
            if replace_flag:
                print(f"Warning: Class {wnid} has only {len(paths)} images, "
                      f"but need {required_imgs}. Using replacement.")

            selected_paths = np.random.choice(paths, required_imgs, replace=replace_flag)

            for path in selected_paths[:self.num_shot]:
                image = Image.open(path).convert("RGB")
                support_x.append(self.transform(image))
                support_y.append(i)

            for path in selected_paths[self.num_shot:]:
                image = Image.open(path).convert("RGB")
                query_x.append(self.transform(image))
                query_y.append(i)

        return (torch.stack(support_x), torch.LongTensor(support_y),
                torch.stack(query_x), torch.LongTensor(query_y),
                episode_class_names)


class CifarFs(Dataset):


    def __init__(self, mode, root, num_ways, num_shots, **kwargs):
        root_dirs = [osp.join(root, 'cifar_fs'), osp.join(root, 'cifar-fs')]
        self.root = next((d for d in root_dirs if osp.exists(d)), None)
        if not self.root:
            raise FileNotFoundError(f"CIFAR-FS not found in {root}")

        self.mode = mode
        self.num_way = num_ways
        self.num_shot = num_shots
        self.num_query = kwargs.get('num_query', 15)

        split_file = osp.join(self.root, 'splits', f'{mode}.txt')
        if not osp.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, 'r') as f:
            self.classes = [line.strip() for line in f if line.strip()]
        self.num_classes = len(self.classes)

        data_dirs = [osp.join(self.root, 'data'), osp.join(self.root, 'images')]
        data_root = next((d for d in data_dirs if osp.exists(d)), None)

        self.class_dict = defaultdict(list)
        for class_name in self.classes:
            class_dir = osp.join(data_root, class_name)
            if osp.isdir(class_dir):
                images = [osp.join(class_dir, img) for img in os.listdir(class_dir)
                          if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                self.class_dict[class_name] = images

        self.label_name = {k: k for k in self.classes}
        self.transform = clip_preprocess_from_pil
        print(f"INFO: [CifarFs] {mode} - {self.num_classes} classes, "
              f"{sum(len(v) for v in self.class_dict.values())} images")

    def __len__(self):
        return 10000

    def __getitem__(self, index):
        selected_class_keys = np.random.choice(self.classes, self.num_way, replace=False)
        support_x, support_y, query_x, query_y = [], [], [], []
        episode_class_names = [self.label_name[key] for key in selected_class_keys]

        for i, class_key in enumerate(selected_class_keys):
            paths = self.class_dict[class_key]
            replace = len(paths) < (self.num_shot + self.num_query)
            selected = np.random.choice(paths, self.num_shot + self.num_query, replace=replace)

            for path in selected[:self.num_shot]:
                support_x.append(self.transform(Image.open(path).convert("RGB")))
                support_y.append(i)
            for path in selected[self.num_shot:]:
                query_x.append(self.transform(Image.open(path).convert("RGB")))
                query_y.append(i)

        return (torch.stack(support_x), torch.LongTensor(support_y),
                torch.stack(query_x), torch.LongTensor(query_y),
                episode_class_names)


class Cub200(Dataset):

    def __init__(self, mode, root, num_ways, num_shots, **kwargs):
        self.root = osp.join(root, 'CUB_200_2011')
        self.mode = mode
        self.num_way = num_ways
        self.num_shot = num_shots
        self.num_query = kwargs.get('num_query', 15)

        csv_file = osp.join(self.root, f'{mode}.csv')
        if not osp.exists(csv_file):
            csv_file = osp.join(self.root, 'split', f'{mode}.csv')

        if not osp.exists(csv_file):
            raise FileNotFoundError(f"CSV split file not found: {csv_file}")

        print(f"Loading {mode} data from: {csv_file}")


        class_to_images = defaultdict(list)
        class_labels = set()

        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)

            for row in reader:
                if len(row) < 2:
                    continue

                filename = row[0].strip()
                label = row[1].strip()

                if '.' not in label:
                    continue
                class_id_str, class_name = label.split('.', 1)
                class_id = int(class_id_str)
                class_labels.add((class_id, label))
                img_path = osp.join(self.root, 'images', label, filename)

                if osp.exists(img_path):
                    class_to_images[class_id].append(img_path)
                else:
                    img_path = osp.join(self.root, 'images', filename)
                    if osp.exists(img_path):
                        class_to_images[class_id].append(img_path)

        self.all_class_names = {}
        for class_id, label in class_labels:
            class_name = label.split('.', 1)[1]
            self.all_class_names[class_id] = class_name.replace('_', ' ').lower()

        self.classes = sorted(class_to_images.keys())
        self.class_dict = class_to_images
        self.num_classes = len(self.classes)

        self.transform = clip_preprocess_from_pil
        print(f"INFO: [Cub200CSV] {mode} - {self.num_classes} classes, "
              f"{sum(len(v) for v in self.class_dict.values())} images")

        # 验证：打印前3个类别
        if self.num_classes > 0:
            print("Sample classes:")
            for cls_id in self.classes[:3]:
                print(f"  ID {cls_id}: '{self.all_class_names[cls_id]}' "
                      f"({len(self.class_dict[cls_id])} images)")

    def __len__(self):
        return 10000

    def __getitem__(self, index):
        selected_class_keys = np.random.choice(self.classes, self.num_way, replace=False)
        support_x, support_y, query_x, query_y = [], [], [], []
        episode_class_names = [self.all_class_names[key] for key in selected_class_keys]

        for i, class_key in enumerate(selected_class_keys):
            paths = self.class_dict[class_key]
            selected = np.random.choice(paths, self.num_shot + self.num_query, replace=False)

            for path in selected[:self.num_shot]:
                support_x.append(self.transform(Image.open(path).convert("RGB")))
                support_y.append(i)
            for path in selected[self.num_shot:]:
                query_x.append(self.transform(Image.open(path).convert("RGB")))
                query_y.append(i)

        return (torch.stack(support_x), torch.LongTensor(support_y),
                torch.stack(query_x), torch.LongTensor(query_y),
                episode_class_names)


class MetaDataset:
    def __init__(self, **kwargs):
        raise NotImplementedError("MetaDataset loader needs a custom implementation.")