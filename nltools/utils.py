"""Handy utilities"""

__all__ = ['get_resource_path']

from os.path import dirname, join, pardir, sep as pathsep

def get_resource_path():
	return join(dirname(__file__), 'resources') + pathsep