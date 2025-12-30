#!/usr/bin/env python3
import sys
from ete3 import Tree

# 从 stdin 或命令行参数读取
if len(sys.argv) > 1:
    tree_str = open(sys.argv[1]).read().strip()
else:
    tree_str = sys.stdin.read().strip()

t = Tree(tree_str)
print(t)
