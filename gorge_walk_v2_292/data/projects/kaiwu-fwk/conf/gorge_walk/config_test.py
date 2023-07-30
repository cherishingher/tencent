#!/usr/bin/env python3
# -*- coding:utf-8 -*-

'''
@Project :kaiwu-fwk 
@File    :config_test.py
@Author  :kaiwu
@Date    :2022/12/1 10:37 

'''

import json 
import numpy as np 


with open("/data/projects/kaiwu-fwk/conf/system/map_data/map_4.json", "r") as f:
    d = json.load(f)
    
map_ = np.array(d["Flags"]).reshape(d["Height"], d["Width"])

# 可视化打印map里的内容
def view(map, point=None):
    out = ""
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            if point:
                if point[0] == i and point[1] == j:
                    out = out + "P"
                    continue 
            if map[i, j] == 1:
                out = out + " "
            else:
                out = out + "x"
        
        out = out + "\n"
    print(out)
    return out 

if __name__ == "__main__":
    #(91, 22) (76, 27) (91, 14) (108, 19) (102, 31)
    #                           x: 9685, z: -9834  x: 15442, -12777
    view(map_, (108, 19))