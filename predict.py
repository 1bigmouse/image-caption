import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from model import EncoderCNN, DecoderRNN
from torchvision import transforms
from bulid_vocab import Vocabulary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
    image = Image.open(image_path)

    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize(244),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    with open(args["vocab_path"], 'rb') as f:
        vocab = pickle.load(f)

    encoder = EncoderCNN(args["embed_size"]).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(args["embed_size"], args["hidden_size"], len(vocab), args["numlayers"]).eval()
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder.load_state_dict(torch.load(args["encoder_path"], "cpu"))
    decoder.load_state_dict(torch.load(args["decoder_path"], "cpu"))
    
    image = load_image(args["image_path"], transform)
    image_tensor = image.to(device)
    
    # Generate an caption from the image
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0]          # (1, max_seq_length) -> (max_seq_length)
    
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id.item()]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sampled_caption = sampled_caption[1: -1]
    sentence = ' '.join(sampled_caption)
    
    # print (sentence)
    # image = Image.open(args["image_path"])
    # plt.imshow(np.array(image))
    return sentence
    
if __name__ == "__main__":
    args = {"vocab_path": r"D:\Extend E Python project\DL_design\pkl\vocab.pkl",
            "image_path": r"/root/autodl-tmp/DL/COCO/val2014/COCO_val2014_000000000074.jpg",
            "encoder_path": r"/root/autodl-tmp/DL/model_pth/epoch0encoder.pth",
            "decoder_path": r"/root/autodl-tmp/DL/model_pth/epoch0decoder.pth",
            "embed_size": 256,
            "hidden_size": 512,
            "numlayers": 1}