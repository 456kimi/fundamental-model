# -*- coding: utf-8 -*-
"""Tushare 凭证读取（不含真实密钥）。

读取顺序：
  1. 环境变量 TUSHARE_TOKEN
  2. 同目录 _secrets_local.py（本机私有，不入版本库）

用法：from _secrets import TOKEN
"""

import os

TOKEN = os.environ.get("TUSHARE_TOKEN", "")

try:
    from _secrets_local import TOKEN as _LOCAL_TOKEN  # noqa: F401

    TOKEN = _LOCAL_TOKEN
except ImportError:
    pass

TUSHARE_TOKEN = TOKEN

__all__ = ["TOKEN", "TUSHARE_TOKEN"]
