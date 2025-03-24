import os
import nltk
import torch
import torch.utils.data as data
from PIL import Image
from pycocotools.coco import COCO
import pickle
from bulid_vocab import Vocabulary
from torchvision.transforms import transforms


class CocoDataset(data.Dataset):
    def __init__(self, root, json, vocab, transform=None):
        """
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target
        """
        return:
            image: PIL.Image
            target: 词索引序列，不定长
        """

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """
    Args:
        data: Dataloader批量抓取dataset,这里把每个批次的大小统一。
            - data: list of tuple (image, caption).  
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape 不一样; variable length.
    """
    # 根据caption长度降序排列
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)
    # print(images[0].shape)
    # print(captions[0].shape)
    
    # 堆叠.
    images = torch.stack(images, 0)
    # print(images.shape)
    
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths

    """
    return:
        images: 堆叠后的一批图像
        targets: 取了本批次最长的caption作为max_length, 其余不足的用0填充("<pad>")
        lengths: 每个caption的长度
    """

def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers):

    # COCO caption dataset
    coco = CocoDataset(root=root,
                       json=json,
                       vocab=vocab,
                       transform=transform)
    # 用自定义的collate_fn replace default
    data_loader = data.DataLoader(dataset=coco,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return data_loader

if __name__ == "__main__":
    Args = {"root": r"D:\Extend E Python project\DL_design\COCO\val2014",
            "json": r"D:\Extend E Python project\DL_design\COCO\annotations\captions_val2014.json",
            "vocab": pickle.load(open(r"D:\Extend E Python project\DL_design\pkl\vocab.pkl", "rb")),
            "transform": transforms.Compose([transforms.ToTensor()])}
    dataset = CocoDataset(**Args)
    image0, targe0 = dataset.__getitem__(1)
    image1, targe1 = dataset.__getitem__(2)
    image2, targe2 = dataset.__getitem__(3)
    # print(image.size, target.shape)
    data = [(image0, targe0), (image1, targe1), (image2, targe2)]
    images, targets, lengths = collate_fn(data)
    print(images.shape, targets.shape, lengths)
    