"""Handy utilities"""

__all__ = ['get_resource_path', 'zscore']

from os.path import dirname, join, pardir, sep as pathsep
import pandas as pd

def get_resource_path():
	return join(dirname(__file__), 'resources') + pathsep

def zscore(df):
	""" zscore every column in a pandas dataframe.
		
		Args:
			df: Pandas DataFrame instance
		
		Returns:
			z_data: z-scored pandas DataFrame instance
	"""

	if not isinstance(df, pd.DataFrame):
		raise ValueError("Data is not a Pandas DataFrame instance")
	
	z_df = df.apply(lambda x: (x - x.mean())/x.std())

	return z_df
