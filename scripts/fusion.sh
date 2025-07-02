#!/bin/bash
nohup python 2_fusion_diy.py --mode fuse --in_dir /data_new2/shizhen/JustTest/Aidite_Crown_Dataset_sixth_align/tooth_crown \
--depth_dir /data_new2/shizhen/JustTest/Aidite_Crown_Dataset_sixth_align/tooth_crown_depth \
--out_dir /data_new2/shizhen/JustTest/Aidite_Crown_Dataset_sixth_align/tooth_crown_watertight \
--resolution 512 > fuse_output.log 2>&1 &
