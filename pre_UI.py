import tkinter as tk
from tkinter import filedialog
from predict import main, load_image
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageTk
from model import EncoderCNN, DecoderRNN
from torchvision import transforms
from bulid_vocab import Vocabulary


def choose():
    global img, image_path
    image_path = filedialog.askopenfilename()
    img = Image.open(image_path)
    img = ImageTk.PhotoImage(img)
    canvas.create_image(250, 200, image=img)


def recognition(image_path):
    args = {"vocab_path": r"D:\Extend E Python project\DL_design\pkl\vocab.pkl",
            "image_path": image_path,
            "encoder_path": r"D:\Extend E Python project\DL_design\model_pth\epoch4encoder.pth",
            "decoder_path": r"D:\Extend E Python project\DL_design\model_pth\epoch4decoder.pth",
            "embed_size": 256,
            "hidden_size": 512,
            "numlayers": 1}
    sentence = main(args)
    var.set(f"{sentence}")


root = tk.Tk()
root.geometry("800x500")
root.title("Generate Image Caption")

canvas = tk.Canvas(master=root, width=500, height=400)
canvas.pack()

var = tk.StringVar()
text1 = tk.Label(root, textvariable=var, bg="yellow", width=60, height=1)
text1.pack()

button2 = tk.Button(master=root, text="choose image ", width=50, height=1, command=choose)
button2.pack()

button1 = tk.Button(master=root, text="describe it ", width=50, height=1,  command=lambda: recognition(image_path))
button1.pack()

root.mainloop()
