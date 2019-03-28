# -*- coding: utf-8 -*-
"""Function group to select file path in GUI."""

import os
import tkinter
import tkinter.filedialog

def file_select(initialdir=__file__):
    root = tkinter.Tk()
    root.withdraw()
    fTyp = [("", "*")]
    iDir = os.path.abspath(os.path.dirname(initialdir))
    file_gui = tkinter.filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)





fld = tkinter.filedialog.askdirectory(initialdir = iDir) 
# 処理ファイル名の出力
print(file)