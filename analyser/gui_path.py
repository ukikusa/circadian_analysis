# -*- coding: utf-8 -*-
"""Function group to select file path in GUI."""

import os
import tkinter
import tkinter.filedialog


def file_select(initialdir=__file__, file_type=[("", "*")]):
    """File select gui."""
    root = tkinter.Tk()
    root.withdraw()
    fTyp = [("", "*")]
    iDir = os.path.abspath(os.path.dirname(initialdir))
    file_path = tkinter.filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)
    return file_path


def dir_select(initialdir=__file__):
    """Folder select gui."""
    root = tkinter.Tk()
    root.withdraw()
    iDir = os.path.abspath(os.path.dirname(initialdir))
    dir_path = tkinter.filedialog.askdirectory(initialdir=iDir)
    return dir_path
