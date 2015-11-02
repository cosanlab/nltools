"""Handy utilities"""

__all__ = ['get_resource_path']

from os.path import dirname, join, pardir, sep as pathsep
import pandas as pd

def get_resource_path():
	return join(dirname(__file__), 'resources') + pathsep

