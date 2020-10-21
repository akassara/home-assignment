import skimage
import os
import torch
from torch.utils.data import Dataset
import albumentations as  A
from Preprocessing.utils import  simple_preprocess


class FoodDataset(Dataset):
    """Foodvisor tomatoe detection dataset."""

    def __init__(self, image_dir, info_df, input_size=(300, 300), transform=None, weights = [1,10]):
        """
        Args:
            info_df (Dataframe): Dataframe of the image paths and annotations.
            image_dir (string): Directory with all the images
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.info_df = info_df
        self.image_dir = image_dir
        self.input_size = input_size
        self.transform = transform
        self.weights = weights

    def __len__(self):
        return len(self.info_df)

    def load_image(self, idx):
        """Generate an image from the specs of the given image ID.

        """
        image_id = self.info_df.loc[idx, "image_path"]
        img_name = os.path.join(self.image_dir, image_id)
        image = skimage.io.imread(img_name)

        return image

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_path = self.info_df.loc[idx, "image_path"]
        img_name = os.path.join(self.image_dir, image_path)
        image = skimage.io.imread(img_name)

        labels = self.info_df.loc[idx, "is_tomato"]

        bboxes = self.info_df.loc[idx, "bbox"]
        resize_transform = A.Resize(self.input_size[0], self.input_size[1])(image=image)
        image = resize_transform['image']

        if self.transform:
            data = {"image": image, "bboxes": bboxes, 'class_labels': labels}
            augmented = self.transform(**data)
            image = augmented['image']
            bboxes = augmented['bboxes']
            labels = augmented['class_labels']
        # Normalization and converting everything to Tensor
        simple_transform = simple_preprocess()(image=image)
        image = simple_transform['image']
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.Tensor([idx])

        # Use the COCO template for targets to be able to evaluate the model with COCO API
        area = bboxes[:, 2] * bboxes[:, 3]
        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
        target = {"boxes": bboxes, "labels": labels, "image_id": image_id, "area": area,
                  "iscrowd": torch.as_tensor([0], dtype=torch.int64)}

        return image, target , self.weights[2 in target['labels']]