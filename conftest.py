# -*- coding: utf-8 -*-
"""测试配置文件：把 src 加入导入路径，保证从仓库根目录即可跑 pytest。"""
import os
import sys

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "src"),
)