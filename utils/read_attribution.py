#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:read_attribution.py
# author:ZCJ
# datetime:2025-07-28 21:35
# software: PyCharm

"""
this is function  description 
"""

# import module your need
import numpy as np

data = {
    'orig_fg': np.array([[0., 0., 0., 0.],
                         [0., 0., 0., 0.],
                         [0., 0., 0., 0.],
                         [0., 1., 1., 0.],
                         [1., 1., 1., 0.],
                         [0., 1., 1., 0.]]),
    'mainstream_fg': np.array([[0., 0., 0., 0.],
                               [0., 0., 0., 0.],
                               [0., 0., 0., 0.],
                               [0., 0., 1., 1.],
                               [1., 1., 0., 0.],
                               [0., 1., 0., 1.]]),
    'tabloid_fg': np.array([[0., 0., 0., 0.],
                            [0., 0., 0., 0.],
                            [0., 0., 0., 0.],
                            [0., 0., 1., 0.],
                            [1., 1., 1., 0.],
                            [0., 1., 1., 0.]]),
    'classes': ['Lack of credible sources', 'False or misleading information', 'Biased opinion', 'Inconsistencies with reputable sources']
}

if __name__ == '__main__':
    # 查看原始文章第一个标签对应的归因问题
    first_orig_labels = data['orig_fg'][0]
    for i, label in enumerate(first_orig_labels):
        if label == 1:
            print(f"原始文章第一篇存在 {data['classes'][i]} 的问题。")

    # 查看主流风格文章第二个标签对应的归因问题
    second_mainstream_labels = data['mainstream_fg'][1]
    for i, label in enumerate(second_mainstream_labels):
        if label == 1:
            print(f"主流风格文章第二篇存在 {data['classes'][i]} 的问题。")