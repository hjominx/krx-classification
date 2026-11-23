"""Puts the code folders of src/ (common, direction, volatility, report) on the import path, so that a script in one
folder can import modules from the others. Every script imports it first; the four copies are identical."""
import os, sys

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ('common', 'direction', 'volatility', 'report'):
    _p = os.path.join(_SRC, _d)
    if _p not in sys.path: sys.path.append(_p)
