# -*-coding: utf-8 -*-
"""移動補正GUIでファイル選択．"""

import os
import sys
import tkinter as tk
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import analyser.gui_path as gui_path
from analyser.transform import frond_transform

class CheckdirsGui:
    """回転確認のダイアルボックスを出す"""

    def __init__(self, calc_folder):
        """a."""
        self.parent_dir, self.calc_dir = os.path.split(calc_folder)
        self.move_folder, self.move_dir = [], []
        self.end = True
        self.tki = tk.Tk()
        self.tki.geometry()
        self.tki.title("フォルダの選択")
        self.var = tk.StringVar()
        text = "計算用画像のdir: " + str(self.calc_dir) + '\n' + "移動用画像のdir: " + "＆".join(self.move_dir)
        self.var.set(text)
        self.label = tk.Label(self.tki, text=text)
        self.label.grid(column=0, row=0)
        self.tki.update_idletasks()

    def dirs(self):
        """a."""
        while self.end:
            def ok_btn():
                self.tki.destroy()
                self.tki.quit()
                self.end = False

            def other_btn():
                self.move_folder.append(gui_path.dir_select(initialdir=self.parent_dir))
                self.move_dir.append(os.path.basename(self.move_folder[-1]))
                text = ("計算用画像のdir: " + str(self.calc_dir) + '\n' + "移動用画像のdir " + "＆".join(self.move_dir))
                self.label = tk.Label(self.tki, text=text)
                self.label.grid(column=0, row=0)
                self.tki.update_idletasks()

            tk.Button(self.tki, text='OK', command=ok_btn).grid(column=0, row=1)
            tk.Button(self.tki, text='append folder', command=other_btn).grid(column=1, row=1)
            self.tki.mainloop()
        return self.move_folder


calc_folder = gui_path.dir_select(initialdir=__file__)
check_dir = CheckdirsGui(calc_folder)
other_folder_list = check_dir.dirs()
frond_transform(parent_directory="", calc_folder=calc_folder, other_folder_list=other_folder_list, motionType=1)
