# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))  # 让 Sphinx 能找到 oak_deepseek 模块

# -- Project information -----------------------------------------------------
project = 'Oak DeepSeek'
copyright = '2026, WyxGenius'
author = 'WyxGenius'
release = '0.3.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',      # 从 docstring 自动生成文档
    'sphinx.ext.napoleon',     # 支持 :param 和 :return 风格
    'sphinx.ext.viewcode',     # 可选，添加源码链接
]

templates_path = ['_templates']
exclude_patterns = []

# 设置语言为简体中文
language = 'zh_CN'

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'
html_static_path = ['_static']

autodoc_default_options = {'members': True, 'undoc-members': True, 'show-inheritance': True, 'no-value': True}