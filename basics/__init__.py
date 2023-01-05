# -*- coding: utf-8 -*-
"""
tsma.basics
"""

import os
import sys

root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

from tsma.basics.transfers import sep, join, decompose_mainname, auto_dct_groups
