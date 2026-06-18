"""Backward-compatible setup.py for older pip/setuptools."""
import sys
from setuptools import setup

if sys.version_info < (3, 8):
    sys.exit("runeextract requires Python >= 3.8")

setup()
