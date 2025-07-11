#!/bin/bash

# 1. 采集数据
python orbbec/capture_orbbec.py --output_dir orbbec/my_data

# 2. 推理
python demo1.py --checkpoint_path checkpoints/checkpoint-rs.tar