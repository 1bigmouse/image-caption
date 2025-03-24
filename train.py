import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from data_loader import get_loader
from model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from tqdm import tqdm
from bulid_vocab import Vocabulary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary 
    with open(args["vocab_path"], 'rb') as f:
        vocab = pickle.load(f)
    
    # Build data loader
    train_data_loader = get_loader(args["train_image_dir"], args["train_caption_path"], vocab, 
                             transform, args["batch_size"],
                             shuffle=True, num_workers=args["num_workers"]) 
    val_data_loader = get_loader(args["val_image_dir"], args["val_caption_path"], vocab,
                                 transform, args["batch_size"],
                                 shuffle=True, num_workers=args["num_workers"])

    # models
    encoder = EncoderCNN(args["embed_size"]).to(device)
    decoder = DecoderRNN(args["embed_size"], args["hidden_size"], len(vocab), args["num_layers"]).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args["learning_rate"])
    
    # Train the models
    total_step = len(train_data_loader)
    for epoch in tqdm(range(args["num_epochs"])):
        encoder.train(), decoder.train()
        print("train start ......")
        for i, (images, captions, lengths) in enumerate(tqdm(train_data_loader, total=total_step)):
            
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            if i % args["print_step"] == 0:
                print(f'Epoch [{epoch}/{args["num_epochs"]}], Step [{i}/{total_step}], Loss: {loss.item()}')
            
        encoder.eval(), decoder.eval()
        print("val start ......")
        with torch.no_grad():
            for i, (images, captions, lengths) in enumerate(val_data_loader):
                images = images.to(device)
                captions = captions.to(device)
                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
                        
                features = encoder(images)
                outputs = decoder(features, captions, lengths)
                
                loss = criterion(outputs, targets)
                if i % args["print_step"] == 0:
                    print(f'Epoch [{epoch}/{args["num_epochs"]}], Step [{i}/{total_step}], Loss: {loss.item()}')
                    break                       
        
        print("save model ......")
        torch.save(encoder.state_dict(), f"D:\Extend E Python project\DL_design\model_pth\epoch{epoch}encoder.pth")
        torch.save(decoder.state_dict(), f"D:\Extend E Python project\DL_design\model_pth\epoch{epoch}decoder.pth")
        print("model saved")
        
        
if __name__ == "__main__":
    args = {"vocab_path": r"D:\Extend E Python project\DL_design\pkl\vocab.pkl",
            "train_image_dir": r"D:\Extend E Python project\DL_design\COCO\train2014",
            "val_image_dir": r"D:\Extend E Python project\DL_design\COCO\val2014",
            "train_caption_path": r"D:\Extend E Python project\DL_design\COCO\annotations\captions_train2014.json",
            "val_caption_path": r"D:\Extend E Python project\DL_design\COCO\annotations\captions_val2014.json",
            "batch_size": 64,
            "num_workers": 2,
            "embed_size": 256,
            "hidden_size": 512,
            "numlayers": 1,
            "learning_rate": 1e-3,
            "num_epochs": 5,
            "print_step": 1000}