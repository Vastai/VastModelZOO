# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import argparse

def create_filelist(folder_path, output_file):
    """
    读取文件夹中的所有jpg文件，并将文件名写入文本文件
    
    Args:
        folder_path: 要读取的文件夹路径
        output_file: 输出文本文件的路径
    """
    try:
        # 检查文件夹是否存在
        if not os.path.exists(folder_path):
            print(f"错误：文件夹 '{folder_path}' 不存在！")
            return
        
        # 获取所有jpg文件（不区分大小写）
        jpg_files = [f for f in os.listdir(folder_path) 
                    if f.lower().endswith('.jpg')]
        
        # 写入文本文件
        with open(output_file, 'w', encoding='utf-8') as f:
            for filename in jpg_files:
                f.write(filename + '\n')
        
        print(f"文件名已成功写入到 '{output_file}'")
        
    except Exception as e:
        print(f"发生错误：{e}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='创建文件夹中jpg文件的文件列表')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='包含jpg图片的输入文件夹路径')
    parser.add_argument('--output_file', type=str, required=True,
                      help='输出文件列表的文本文件路径')
    return parser.parse_args()

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    # 调用函数
    create_filelist(args.input_dir, args.output_file)
