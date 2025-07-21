class Brain_Collection:
    """A container for neuroimaging results with slicing semantics to manage collections of Brain_Data objects.

    Internally represents data as a list of dictionaries. Each dictionary represents a collection of Brain_Data objects with the same set of keys. This makes it easy to work with the output of multiple calls to methods like `Brain_Data.regress()` together.

    # Given data like this:
    results = [
        {
            "z_score": sim_brain_data, # 8 x num_voxels
            "t": sim_brain_data.copy(), # 8 x num_voxels
            "p": sim_brain_data.copy(), # 8 x num_voxels
            "beta": sim_brain_data.copy(), # 8 x num_voxels
            "se": sim_brain_data.copy(), # 8 x num_voxels
            "rsquared": sim_brain_data.copy()[0],  # 1 x num_voxels
            "labels": [  # 8 regressors
                "bottle",
                "cat",
                "chair",
                "face",
                "house",
                "scissors",
                "shoe",
                "scrambledpix",
            ],
        },
        {
            "z_score": sim_brain_data, # 8 x num_voxels
            "t": sim_brain_data.copy(), # 8 x num_voxels
            "p": sim_brain_data.copy(), # 8 x num_voxels
            "beta": sim_brain_data.copy(), # 8 x num_voxels
            "se": sim_brain_data.copy(), # 8 x num_voxels
            "rsquared": sim_brain_data.copy()[0],  # 1 x num_voxels
            "labels": [  # 8 regressors
                "bottle",
                "cat",
                "chair",
                "face",
                "house",
                "scissors",
                "shoe",
                "scrambledpix",
            ],
        },
    ]

    Supports a variety of indexing:
    - results['key'] # gets dict-key across all list items
    - results[numeric-index] # get nth list item
    - results[numeric-index, 'key'] # get nth list item's dict-key
    - results[numeric-index, :, 'label'] # get nth list item's dict-key, sliced into by label
    - results[numeric-index, :, numeric-index] # get nth list item's dict-key, sliced into by numeric-index
    - results[numeric-index, 'key', 'label'] # get nth list item's dict-key, sliced into by label
    - results[numeric-index, 'key', numeric-index] # get nth list item's dict-key, sliced into by numeric-index
    - results[start:stop:step, ['key1', 'key2'...], ['label1', 'label2', 'label3'...]] # full possibilities using labels for last dimension
    - results[start:stop:step, ['key1', 'key2'...], start:stop:step] # full possibilities using numeric-index for last dimension

    Args:
        data (dict | list[dict] | None): A dictionary or list of dictionaries
            with identical keys. If a single dict is provided, it's wrapped in a list.

    Examples:
        >>> # Create from GLM results
        >>> all_results = Brain_Collection()
        >>> all_results.append(brain_data.regress())
        >>>
    """

    def __init__(self, data: dict | list | None = None):
        if data is None:
            self.data = []
            self._keys = tuple()
        else:
            self.data, self._keys = self._validate_input(data)

    def __repr__(self):
        if not self.data:
            return "Brain_Collection(empty)"
        return f"Brain_Collection(n_items={len(self.data)}, keys={list(self._keys)})"

    def _validate_input(self, data):
        """Validate and normalize input data.

        Args:
            data: A dictionary or list of dictionaries

        Returns:
            tuple: (normalized_data, keys)
        """
        # Wrap single dict in a list
        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            raise TypeError(f"Data must be a dict or list of dicts, not {type(data)}")

        if not data:
            return [], tuple()

        # Validate all dicts have the same keys
        first_keys = set(data[0].keys())
        for i, d in enumerate(data):
            if not isinstance(d, dict):
                raise TypeError(f"Item {i} must be a dict, not {type(d)}")
            if set(d.keys()) != first_keys:
                raise ValueError(
                    f"All dictionaries must have the same keys. "
                    f"Item {i} has keys {set(d.keys())} but expected {first_keys}"
                )

        return data, tuple(sorted(first_keys))

    def append(self, other):
        """Append another Brain_Collection or dict to this collection.

        Args:
            other (Brain_Collection | dict): Data to append. If a dict, must have
                the same keys as existing data. If a Brain_Collection, must have
                the same keys.
        """
        if isinstance(other, dict):
            # Validate keys match
            if self._keys and set(other.keys()) != set(self._keys):
                raise ValueError(
                    f"Dictionary keys {set(other.keys())} don't match "
                    f"collection keys {set(self._keys)}"
                )
            self.data.append(other)
            if not self._keys:
                self._keys = tuple(sorted(other.keys()))

        elif isinstance(other, Brain_Collection):
            if not other.data:
                return  # Nothing to append

            # Validate keys match
            if self._keys and other._keys != self._keys:
                raise ValueError(
                    f"Brain_Collection keys {other._keys} don't match "
                    f"this collection's keys {self._keys}"
                )

            # Extend with the other collection's data
            self.data.extend(other.data)
            if not self._keys:
                self._keys = other._keys
        else:
            raise TypeError(
                f"Can only append dict or Brain_Collection, not {type(other)}"
            )

    def __getitem__(self, index):
        """Flexible indexing support.

        Args:
            index: Can be:
                - str: Returns list of values for that key across all items
                - int: Returns entire dict for that item
                - tuple(int, str): Returns specific value
                - tuple(str, str): Label-based indexing using regressor name and field
                - tuple(list[str], str): Multi-label indexing with single field
                - tuple(list[str], list[str]): Multi-label and multi-field indexing
                - tuple(list[int], str): Multiple indices with single field
                - tuple(list[int], list[str]): Multiple indices with multiple fields
                - slice: Returns new Brain_Collection with sliced items
                - tuple(slice, str): Returns list of values for key from sliced items
                - tuple(slice, list[str]): Returns list of dicts with multiple keys from sliced items
        """
        if isinstance(index, str):
            # Get all values for this key
            return self._get_key_values(index)

        elif isinstance(index, int):
            # Get entire dict for this item
            return self.data[index]

        elif isinstance(index, slice):
            # Return new collection with sliced data
            return Brain_Collection(self.data[index])

        elif isinstance(index, tuple) and len(index) == 2:
            item_idx, key = index

            if isinstance(item_idx, int):
                # Get specific value
                return self.data[item_idx][key]

            elif isinstance(item_idx, str):
                # Label-based indexing: ('face', 'z_score')
                return self._get_regressor_values(item_idx, key)

            elif isinstance(item_idx, list):
                # Check if list contains integers or strings
                if all(isinstance(idx, int) for idx in item_idx):
                    # List of integer indices: [0, 1], 'beta' or [0, 1], ['beta', 't']
                    if isinstance(key, str):
                        # Single key with multiple indices
                        result = [self.data[i][key] for i in item_idx]
                        return self._maybe_concatenate_brain_data(result)
                    elif isinstance(key, list):
                        # Multiple keys with multiple indices
                        return self._get_multi_index_multi_key_values(item_idx, key)
                    else:
                        raise TypeError(
                            f"With list of indices, key must be str or list, not {type(key)}"
                        )
                elif all(isinstance(idx, str) for idx in item_idx):
                    # Multi-label indexing: (['face', 'house'], 'beta') or (['face', 'house'], ['beta', 't'])
                    if isinstance(key, str):
                        # Single key with multiple labels
                        return self._get_multi_regressor_values(item_idx, [key])
                    elif isinstance(key, list):
                        # Multiple keys with multiple labels
                        return self._get_multi_regressor_values(item_idx, key)
                    else:
                        raise TypeError(
                            f"With list of labels, key must be str or list, not {type(key)}"
                        )
                else:
                    raise TypeError("List indices must be all integers or all strings")

            elif isinstance(item_idx, slice):
                # Get values for key from sliced items
                sliced_data = self.data[item_idx]
                if isinstance(key, str):
                    # Single key with slice
                    if len(sliced_data) == 1:
                        return sliced_data[0][key]
                    result = [d[key] for d in sliced_data]
                    return self._maybe_concatenate_brain_data(result)
                elif isinstance(key, list):
                    # Multiple keys with slice
                    indices = list(range(len(self.data)))[item_idx]
                    return self._get_multi_index_multi_key_values(indices, key)
                else:
                    raise TypeError(
                        f"With slice, key must be str or list, not {type(key)}"
                    )

            else:
                raise TypeError(
                    f"First index must be int, str, list, or slice, not {type(item_idx)}"
                )

        else:
            raise TypeError(f"Invalid index type: {type(index)}")

    def _get_key_values(self, key):
        """Get all values for a given key across items.

        Returns single value if only one item, otherwise a list.
        """
        if key not in self._keys:
            raise KeyError(f"Key '{key}' not found. Available keys: {list(self._keys)}")

        if len(self.data) == 0:
            return []
        elif len(self.data) == 1:
            return self.data[0][key]
        else:
            result = [d[key] for d in self.data]
            # Check if we should concatenate Brain_Data singletons
            return self._maybe_concatenate_brain_data(result)

    def _get_regressor_values(self, regressor_name, field_name):
        """Get values for a specific regressor from Brain_Data objects.

        Args:
            regressor_name (str): Name of the regressor (e.g., 'face')
            field_name (str): Name of the field containing Brain_Data (e.g., 'z_score')

        Returns:
            Brain_Data or list of Brain_Data: Sliced Brain_Data for the specific regressor
        """
        # Check if 'regressors' field exists
        if "regressors" not in self._keys:
            raise KeyError(
                "No 'regressors' field found in the collection. "
                "Label-based indexing requires a 'regressors' field."
            )

        # Check if the requested field exists
        if field_name not in self._keys:
            raise KeyError(
                f"Field '{field_name}' not found. Available fields: {list(self._keys)}"
            )

        # Process each item in the collection
        results = []
        for i, item in enumerate(self.data):
            regressors = item["regressors"]

            # Find the index of the regressor
            try:
                regressor_idx = list(regressors).index(regressor_name)
            except ValueError:
                raise ValueError(
                    f"Regressor '{regressor_name}' not found in item {i}. "
                    f"Available regressors: {list(regressors)}"
                )

            # Get the Brain_Data object and slice it
            brain_data = item[field_name]
            # Assuming Brain_Data objects support indexing
            sliced_data = brain_data[regressor_idx]
            results.append(sliced_data)

        # Return single item if only one, otherwise return list
        if len(results) == 1:
            return results[0]
        else:
            # Check if we should concatenate Brain_Data singletons
            return self._maybe_concatenate_brain_data(results)

    def _get_multi_regressor_values(self, regressor_names, field_names):
        """Get values for multiple regressors and/or fields from Brain_Data objects.

        Args:
            regressor_names (list[str]): Names of the regressors (e.g., ['face', 'house'])
            field_names (list[str]): Names of the fields containing Brain_Data (e.g., ['beta', 't'])

        Returns:
            dict or list[dict]: Dictionary mapping (regressor, field) tuples to Brain_Data objects.
                If only one item in collection, returns dict. Otherwise returns list of dicts.
        """
        # Check if 'regressors' field exists
        if "regressors" not in self._keys:
            raise KeyError(
                "No 'regressors' field found in the collection. "
                "Label-based indexing requires a 'regressors' field."
            )

        # Check if all requested fields exist
        for field_name in field_names:
            if field_name not in self._keys:
                raise KeyError(
                    f"Field '{field_name}' not found. Available fields: {list(self._keys)}"
                )

        # Process each item in the collection
        all_results = []

        for i, item in enumerate(self.data):
            regressors = item["regressors"]
            item_results = {}

            # Process each regressor
            for regressor_name in regressor_names:
                # Find the index of the regressor
                try:
                    regressor_idx = list(regressors).index(regressor_name)
                except ValueError:
                    raise ValueError(
                        f"Regressor '{regressor_name}' not found in item {i}. "
                        f"Available regressors: {list(regressors)}"
                    )

                # Get values for each field
                for field_name in field_names:
                    brain_data = item[field_name]
                    sliced_data = brain_data[regressor_idx]
                    item_results[f"{regressor_name}_{field_name}"] = sliced_data

            all_results.append(item_results)

        # Return single dict if only one item, otherwise return list of dicts
        if len(all_results) == 1:
            return all_results[0]
        else:
            return all_results

    def _get_multi_index_multi_key_values(self, indices, keys):
        """Get values for multiple indices and multiple keys.

        Args:
            indices (list[int]): List of integer indices
            keys (list[str]): List of field names

        Returns:
            list[dict]: List of dictionaries, one per index, with requested keys
        """
        # Check if all requested keys exist
        for key in keys:
            if key not in self._keys:
                raise KeyError(
                    f"Key '{key}' not found. Available keys: {list(self._keys)}"
                )

        # Build result for each index
        results = []
        for idx in indices:
            item_result = {}
            for key in keys:
                item_result[key] = self.data[idx][key]
            results.append(item_result)

        return results

    def _maybe_concatenate_brain_data(self, result):
        """Check if result is a list of Brain_Data singletons and concatenate if so.

        Args:
            result: The result from a slicing operation

        Returns:
            Either the original result or a concatenated Brain_Data object
        """
        # Import here to avoid circular imports
        from .brain_data import Brain_Data

        # Only process lists
        if not isinstance(result, list):
            return result

        # Check if all items are Brain_Data objects
        if not all(
            hasattr(item, "__class__") and item.__class__.__name__ == "Brain_Data"
            for item in result
        ):
            return result

        # Check if all Brain_Data objects are singletons (1D data)
        try:
            if all(
                hasattr(item, "data")
                and hasattr(item.data, "ndim")
                and item.data.ndim == 1
                for item in result
            ):
                # Concatenate into a single Brain_Data object
                return Brain_Data(result)
        except Exception:
            # If anything goes wrong, return original result
            pass

        return result

    def __len__(self):
        """Number of items in the collection."""
        return len(self.data)

    def __iter__(self):
        """Iterate over items (dictionaries)."""
        return iter(self.data)

    def __contains__(self, key):
        """Check if a key exists in the collection."""
        return key in self._keys

    def keys(self):
        """Return the keys available in this collection."""
        return list(self._keys)

    @property
    def shape(self):
        """Return shape as (n_items, n_keys)."""
        return (len(self.data), len(self._keys))

    def get(self, key, default=None):
        """Get values for a key with a default if key doesn't exist."""
        try:
            return self._get_key_values(key)
        except KeyError:
            return default

    def __getattr__(self, name):
        """Enable attribute-style access to keys (e.g., collection.beta)."""
        if name in self._keys:
            return self._get_key_values(name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )
