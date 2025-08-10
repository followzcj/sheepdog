#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:deldata.py
# author:ZCJ
# datetime:2025-08-04 19:09
# software: PyCharm

"""
this is function  description 
"""

# import module your need
import os


def clear_files_in_subfolders(folder_path):
    if not os.path.exists(folder_path):
        return

    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            for file in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)


if __name__ == "__main__":
    target_folder = '../data'
    os.remove('data.pkl')
    clear_files_in_subfolders(target_folder)

