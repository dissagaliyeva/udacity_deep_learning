import re
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
# from torchsampler import ImbalancedDatasetSampler

import os
import splitfolders


def create_loaders(n_batch, train_path=None, test_path=None,
                   own=False, cuda=True,
                   single=False, img_path=None):
    """"
    Returns three loaders (train, validation, test) along with the dataset containing transformed images.
        Parameters:
            n_batch (int): Size of a batch
            train_path (str): Path to the train folder
            test_path (str): Path to the test folder
            single (bool): Single image prediction
            img_path (str): Path to a single image

        Returns: data loaders and dataset if single=False, otherwise transformations
    """

    # define the transformations for ground-up architecture
    own_transform = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomRotation((-10, 10)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),

        'valid_test': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    # define the transformations for VGG16 architecture
    transfer_transform = {
        'train': transforms.Compose([
            transforms.Resize((226, 226)),
            transforms.CenterCrop(224),
            transforms.RandomRotation((-10, 10)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),

        'valid_test': transforms.Compose([
            transforms.Resize((226, 226)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    # assign the transformation
    transform = transfer_transform

    # return a single transformation for STEP 3: single image transformations
    if single:
        ts = transform['test'](Image.open(img_path)).unsqueeze_(0)
        return ts.cuda() if cuda else ts

    # re-define transform if it's custom architecture
    if own:
        transform = own_transform

    # create an easier split of train and validation folders
    if not os.path.isdir('output'):
        _ = splitfolders.ratio(train_path, output='output', ratio=(.8, .2), group_prefix=None)

    # read the files and assign to their appropriate values
    train_holder = datasets.ImageFolder('output/train', transform=transform['train'])
    validation_holder = datasets.ImageFolder('output/val', transform=transform['valid_test'])
    test_holder = datasets.ImageFolder(test_path, transform=transform['valid_test'])
    dataset = train_holder  # holder to display images

    # define the loaders
    train_loader = torch.utils.data.DataLoader(train_holder, batch_size=n_batch, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_holder, batch_size=n_batch, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_holder, batch_size=n_batch, shuffle=True)

    return train_loader, validation_loader, test_loader, dataset


# create dictionary that stores vocabulary
class Dictionary:
    """
    A class to represent a dictionary.

        Attributes:
            dataset: ImageFolder that stores images from training folder

        Methods:
            create_inverse():   Creates the inverse of the dictionary. Stores "int: class" pairs
            simple_print(idx):  Shows first N integers and their respective classes
            get_item(item):     Returns the class name at specific index
            get_content(index): Returns string representation of a list of indices
    """

    def __init__(self, dataset):
        """
        Instantiates the inverse dictionary.

        Parameters:
            dataset: ImageFolder that stores images from training folder
        """
        self.dataset = dataset
        self.inverse_dict = self.create_inverse()

    def create_inverse(self) -> dict:
        """
        Creates the inverse of the dictionary. Stores "int: class" pairs

        Returns: An inverse dictionary (int: class)
        """
        return dict((v, k) for k, v in self.dataset.class_to_idx.items())

    def simple_print(self, idx=50):
        """
        Shows first N integers and their respective classes.

        Parameters:
            idx (int): Number of indices to show

        Returns: None
        """
        # show all classes from training folder
        print('\t\t\t\t\t\tClasses & Indexes')
        for i, v in enumerate(self.dataset.class_to_idx.values()):
            if i == idx: break
            print(f'{v}:\t{self.get_content(v)}')

    def get_item(self, item: int) -> str:
        """
        Returns a class name from a dictionary.

        Parameters:
            item (int): Index value to lookup

        Returns: Class name
        """
        return self.inverse_dict[item]

    def get_content(self, index) -> str:
        """
        Gets the indices and outputs a beautiful representation of class names.

        Parameters:
            index (list or int): List or index values to lookup

        Returns: string representation separated by commas
        """

        # remove leading digits and underscores
        make_prettier = lambda x: ' '.join(re.findall('[A-Za-z]+', self.get_item(x)))

        # check if it's a single index
        if type(index) == int:
            return make_prettier(index)

        # return comma-separated representation
        return ', '.join([make_prettier(x) for x in index])


def visualize(loader, dictionary, single=False):
    """
    Create single and batch visualizations.

    Parameters:
        loader:             Instance of DataLoader to iterate through
        dictionary (class): Previously created Dictionary class
        single (bool):      Show a single image or not. Default is False
    """

    # create converters for images and labels
    convert = lambda x: np.clip(x.numpy().transpose((1, 2, 0)), 0, 1)
    convert_label = lambda x: str(x.item())

    # transform single images and their labels
    def show_single(image, lbl, index=0):
        image = image * 2.0 - 1.0         # unnormalize the image
        image = convert(image[index, :])  # transform the image
        lbl = convert_label(lbl[index])   # get the label from dictionary
        return image, dictionary.get_content(int(lbl))

    # iterate through one or batch of images
    images, labels = next(iter(loader))
    img_len = len(images)

    # show single image
    if single:
        i, l = show_single(images, labels)
        plt.title(l, fontsize=20)
        plt.imshow(i)
    else:
        # create a figure to show img_len batch of images
        fig = plt.figure(figsize=(30, 10))
        fig.tight_layout(pad=3.0)
        fig.suptitle(f'Sample batch of {img_len}', fontsize=40, y=0.55)

        for idx in range(img_len):
            ax = fig.add_subplot(2, int(img_len / 2), idx + 1, xticks=[], yticks=[])
            image, label = show_single(images, labels, idx)
            ax.set_title(label, fontsize=15)
            ax.imshow(image)
    plt.show()
