import os
from PIL import Image
from torch.utils.data import Dataset , DataLoader
from torchvision import transforms
def find_class_by_filename(file_path, target_filename):
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                filename = parts[0]
                class_label = parts[1]

                if filename == target_filename:
                    return class_label
    return None
class TinyImageDataset_train(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_name=[]
        for label, class_dir in enumerate(os.listdir(root_dir)):
            images_path = os.path.join(root_dir,class_dir, 'images')
            if os.path.isdir(images_path):
                for img_name in os.listdir(images_path):
                    img_path = os.path.join(images_path, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(label)
            self.label_name.append(class_dir)
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        image = self.transform(image)
        return image, label
    def get_label_name(self):
        return self.label_name
class TinyImageDataset_test(Dataset):
    def __init__(self, root_dir,labels_name, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels_name=labels_name
        self.labels = []
        images_pate = os.path.join(root_dir, 'images',)
        file_path = os.path.join(root_dir,'val_annotations.txt')
        if os.path.isdir(images_pate):
            for img_name in os.listdir(images_pate):
                self.image_paths.append(os.path.join(images_pate,img_name))
                label_name_temp = find_class_by_filename(file_path, img_name)
                for  label , label_name in enumerate(self.labels_name):
                    if label_name == label_name_temp:
                        self.labels.append(label)
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
if __name__ == '__main__':
    root_dir_train = 'tiny-imagenet-200/train'
    root_dir_val  = 'tiny-imagenet-200/val'
    train_dataset = TinyImageDataset_train(root_dir_train)
    labels_name = train_dataset.get_label_name()
    val_dataset = TinyImageDataset_test(root_dir_val,labels_name )
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=4)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=4)

    for data , label in val_dataloader:
        print(data,label)
