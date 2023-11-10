import datetime
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import serial
import torch
from PIL import ImageTk, Image


# Thread 1
class windows_system(tk.Tk):

    def __init__(self):
        # tao bien dung chung
        super().__init__()
        # tao windows chinh
        self.title("Training Model Mode")
        self.geometry("1280x720")
        self.resizable(width=False, height=False)
        icon = tk.PhotoImage(file=r"D:\Luan Van\images\logo.png")
        self.iconphoto(True, icon)
        background_image = tk.PhotoImage(
            file=r"D:\Luan Van\images\background.png")
        self.background_label = tk.Label(image=background_image)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)



        self.mainloop()


# chay chuong trinh
if __name__ == "__main__":
    windows_system()
