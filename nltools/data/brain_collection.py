class Brain_Collection:
    """A container for neuroimaging results with 3D slicing semantics to manage collections of Brain_Data objects.

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

    Supports 3D indexing with the pattern: collection[position, key, label]
    - position: item index (int, slice, list of ints, or ':')
    - key: dictionary key (str, list of strs, or ':')
    - label: label name (str, int, slice, list, or ':')

    Examples:
    - results['beta'] # All items, single key, all labels
    - results[0] # Single item (full dict)
    - results[0, 'beta'] # Single item, single key, all labels
    - results[:, 'beta', 'face'] # All items, single key, single label
    - results[:, :, 'face'] # All items, all keys, single label
    - results[0, 'beta', 'face'] # Single item, single key, single label
    - results[:2, ['beta', 't']] # Slice items, multiple keys, all labels
    - results[:, ['beta', 't'], ['face', 'house']] # All items, multiple keys, multiple labels

    Args:
        data (dict | list[dict] | None): A dictionary or list of dictionaries
            with identical keys. If a single dict is provided, it's wrapped in a list.
            The 'labels' key is extracted and stored separately if present.

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
            self._labels = None
        else:
            self.data, self._keys, self._labels = self._validate_input(data)

    def __repr__(self):
        if not self.data:
            return "Brain_Collection(empty)"
        label_info = f", labels={self._labels}" if self._labels is not None else ""
        return f"Brain_Collection(n_items={len(self.data)}, keys={list(self._keys)}{label_info})"

    def __eq__(self, other):
        """Check equality between Brain_Collections."""
        if not isinstance(other, Brain_Collection):
            return False

        # Compare basic attributes
        if len(self.data) != len(other.data):
            return False
        if self._keys != other._keys:
            return False
        if self._labels != other._labels:
            return False

        # Compare data contents
        for i, (self_item, other_item) in enumerate(zip(self.data, other.data)):
            for key in self._keys:
                if key not in self_item or key not in other_item:
                    return False
                # Use Brain_Data's equality if available
                self_val = self_item[key]
                other_val = other_item[key]
                try:
                    if self_val != other_val:
                        return False
                except Exception:
                    # If comparison fails, they're not equal
                    return False

        return True

    def _validate_input(self, data):
        """Validate and normalize input data.

        Args:
            data: A dictionary or list of dictionaries

        Returns:
            tuple: (normalized_data, keys, labels)
        """
        if not data:
            return [], tuple(), None

        # Wrap single dict in a list
        if isinstance(data, dict):
            data = [data]
        elif isinstance(data, list):
            if not all(isinstance(d, dict) for d in data):
                raise TypeError("All list elements must be dicts")
        else:
            raise TypeError(f"Data must be a dict or list of dicts, not {type(data)}")

        # Extract labels if present
        labels = None
        processed_data = []

        for i, d in enumerate(data):
            # Create a copy to avoid modifying the original
            d_copy = d.copy()

            # Extract labels if present
            if "labels" in d_copy:
                item_labels = d_copy.pop("labels")
                if labels is None:
                    labels = item_labels
                elif labels != item_labels:
                    raise ValueError(
                        f"All items must have the same labels. "
                        f"Item {i} has labels {item_labels} but expected {labels}"
                    )

            processed_data.append(d_copy)

        # Validate all dicts have the same keys (after removing labels)
        if processed_data:
            first_keys = set(processed_data[0].keys())
            for i, d in enumerate(processed_data):
                if set(d.keys()) != first_keys:
                    raise ValueError(
                        f"All dictionaries must have the same keys. "
                        f"Item {i} has keys {set(d.keys())} but expected {first_keys}"
                    )

        keys = tuple(sorted(first_keys)) if processed_data else tuple()
        return processed_data, keys, labels

    def append(self, other):
        """Append another Brain_Collection or dict to this collection.

        Args:
            other (Brain_Collection | dict): Data to append. If a dict, must have
                the same keys as existing data (excluding 'labels'). If a Brain_Collection,
                must have the same keys and labels.
        """
        if isinstance(other, dict):
            # Process the dict to extract labels
            other_copy = other.copy()
            other_labels = other_copy.pop("labels", None)

            # Validate labels match if we have them
            if self._labels is not None and other_labels is not None:
                if self._labels != other_labels:
                    raise ValueError(
                        f"Labels {other_labels} don't match "
                        f"collection labels {self._labels}"
                    )
            elif self._labels is None and other_labels is not None:
                self._labels = other_labels
            elif self._labels is not None and other_labels is None:
                raise ValueError(
                    "New item must have labels to match existing collection"
                )

            # Validate keys match (excluding labels)
            other_keys = set(other_copy.keys())
            if self._keys and other_keys != set(self._keys):
                raise ValueError(
                    f"Dictionary keys {other_keys} don't match "
                    f"collection keys {set(self._keys)}"
                )

            self.data.append(other_copy)
            if not self._keys:
                self._keys = tuple(sorted(other_keys))

        elif isinstance(other, Brain_Collection):
            if not other.data:
                return  # Nothing to append

            # Validate keys and labels match
            if self._keys and other._keys != self._keys:
                raise ValueError(
                    f"Brain_Collection keys {other._keys} don't match "
                    f"this collection's keys {self._keys}"
                )

            if self._labels is not None and other._labels != self._labels:
                raise ValueError(
                    f"Brain_Collection labels {other._labels} don't match "
                    f"this collection's labels {self._labels}"
                )

            # Extend with the other collection's data
            self.data.extend(other.data)
            if not self._keys:
                self._keys = other._keys
            if self._labels is None:
                self._labels = other._labels
        else:
            raise TypeError(
                f"Can only append dict or Brain_Collection, not {type(other)}"
            )

    def __getitem__(self, index):
        """3D indexing support: collection[position, key, label].

        Args:
            index: Can be:
                - str: Shorthand for [:, key, :]
                - int: Shorthand for [position, :, :]
                - slice: Shorthand for [slice, :, :]
                - tuple with 2 or 3 elements representing (position, key, label)

        Each dimension can be:
            - position: int, slice, list of ints, or ':' (all)
            - key: str, list of strs, or ':' (all)
            - label: str, int, slice, list, or ':' (all)
        """
        # Normalize the index to a 3-tuple
        position, key, label = self._normalize_index(index)

        # Count singleton dimensions (for intelligent flattening)
        is_single_position = isinstance(position, int)
        is_single_key = isinstance(key, str)
        is_single_label = self._is_single_label(label)

        # Get items based on position
        items = self._get_items_by_position(position)

        # If just getting raw items (no key/label filtering)
        if (
            isinstance(key, slice)
            and key == slice(None)
            and isinstance(label, slice)
            and label == slice(None)
        ):
            if isinstance(position, int):
                return items  # Single dict
            else:
                collection = Brain_Collection(items)
                collection._labels = self._labels
                return collection

        # Get data for specified keys
        data_dict = self._get_keys_from_items(items, key)

        # Apply label slicing if needed
        if not (isinstance(label, slice) and label == slice(None)):
            data_dict = self._apply_label_slicing(data_dict, label)

        # Apply intelligent flattening
        # Special handling for the case where key is passed to format_output
        actual_key = key if is_single_key else None
        return self._format_output(
            data_dict,
            is_single_position,
            is_single_key,
            is_single_label,
            actual_key,
            label,
        )

    def _is_single_label(self, label):
        """Check if label index represents a single label."""
        if isinstance(label, (int, str)):
            return True
        if isinstance(label, list) and len(label) == 1:
            return True
        return False

    def _normalize_index(self, index):
        """Normalize index to a 3-tuple (position, key, label)."""
        if isinstance(index, str):
            # Shorthand: collection['key'] -> collection[:, 'key', :]
            return slice(None), index, slice(None)
        elif isinstance(index, int):
            # Shorthand: collection[0] -> collection[0, :, :]
            return index, slice(None), slice(None)
        elif isinstance(index, slice):
            # Shorthand: collection[1:3] -> collection[1:3, :, :]
            return index, slice(None), slice(None)
        elif isinstance(index, tuple):
            if len(index) == 1:
                return self._normalize_index(index[0])
            elif len(index) == 2:
                # collection[position, key] -> collection[position, key, :]
                pos, key = index
                # Normalize each component
                if pos == ":":
                    pos = slice(None)
                if key == ":":
                    key = slice(None)
                return pos, key, slice(None)
            elif len(index) == 3:
                pos, key, label = index
                # Normalize each component
                if pos == ":":
                    pos = slice(None)
                if key == ":":
                    key = slice(None)
                if label == ":":
                    label = slice(None)
                return pos, key, label
            else:
                raise ValueError(
                    "Too many indices. Expected at most 3 (position, key, label)"
                )
        else:
            raise TypeError(f"Invalid index type: {type(index)}")

    def _get_items_by_position(self, position):
        """Get items based on position index."""
        if isinstance(position, slice) and position == slice(None):
            return self.data
        elif isinstance(position, int):
            return self.data[position]
        elif isinstance(position, slice):
            return self.data[position]
        elif isinstance(position, list):
            if not all(isinstance(i, int) for i in position):
                raise TypeError("Position list must contain only integers")
            return [self.data[i] for i in position]
        else:
            raise TypeError(f"Invalid position type: {type(position)}")

    def _get_keys_from_items(self, items, key):
        """Extract specified keys from items."""
        # Handle single item case
        if isinstance(items, dict):
            items = [items]

        if isinstance(key, slice) and key == slice(None):
            # Return all data
            if len(items) == 1:
                return items[0]
            return items
        elif isinstance(key, str):
            # Single key
            if key not in self._keys:
                raise KeyError(
                    f"Key '{key}' not found. Available keys: {list(self._keys)}"
                )
            if len(items) == 1:
                return items[0][key]
            return [item[key] for item in items]
        elif isinstance(key, list):
            # Multiple keys
            for k in key:
                if k not in self._keys:
                    raise KeyError(
                        f"Key '{k}' not found. Available keys: {list(self._keys)}"
                    )
            result = []
            for item in items:
                result.append({k: item[k] for k in key})
            if len(result) == 1:
                return result[0]
            return result
        else:
            raise TypeError(f"Invalid key type: {type(key)}")

    def _apply_label_slicing(self, data, label):
        """Apply label-based slicing to Brain_Data objects."""
        # Handle integer-based slicing when no labels
        if self._labels is None and isinstance(label, (int, slice, list)):
            # Just apply the slicing directly
            return self._apply_direct_slicing(data, label)

        if self._labels is None:
            raise ValueError(
                "No labels found in collection. Cannot perform label-based indexing."
            )

        # Determine label indices
        if isinstance(label, str):
            if label not in self._labels:
                raise ValueError(
                    f"Label '{label}' not found. Available labels: {self._labels}"
                )
            label_indices = [self._labels.index(label)]
        elif isinstance(label, int):
            label_indices = [label]
        elif isinstance(label, slice):
            label_indices = list(range(len(self._labels)))[label]
        elif isinstance(label, list):
            label_indices = []
            for l in label:
                if isinstance(l, str):
                    if l not in self._labels:
                        raise ValueError(
                            f"Label '{l}' not found. Available labels: {self._labels}"
                        )
                    label_indices.append(self._labels.index(l))
                elif isinstance(l, int):
                    label_indices.append(l)
                else:
                    raise TypeError(
                        f"Label list items must be str or int, not {type(l)}"
                    )
        else:
            raise TypeError(f"Invalid label type: {type(label)}")

        return self._apply_direct_slicing(data, label_indices)

    def _apply_direct_slicing(self, data, indices):
        """Apply direct index-based slicing to Brain_Data objects."""
        if isinstance(data, dict):
            # Single item with multiple keys
            result = {}
            for k, v in data.items():
                if hasattr(v, "__getitem__") and hasattr(v, "shape"):
                    # Check if this Brain_Data can be sliced
                    shape = v.shape()
                    if len(shape) > 0 and shape[0] > 1:
                        # Multi-dimensional, can be sliced
                        if isinstance(indices, int):
                            result[k] = v[indices]
                        elif isinstance(indices, list) and len(indices) == 1:
                            result[k] = v[indices[0]]
                        elif isinstance(indices, slice):
                            result[k] = v[indices]
                        else:
                            # Multiple indices
                            sliced_items = [v[i] for i in indices]
                            result[k] = self._maybe_concatenate_brain_data(sliced_items)
                    else:
                        # 1D or single row, return as-is
                        result[k] = v
                else:
                    result[k] = v
            return result
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            # Multiple items
            result = []
            for item in data:
                item_result = {}
                for k, v in item.items():
                    if hasattr(v, "__getitem__") and hasattr(v, "shape"):
                        shape = v.shape()
                        if len(shape) > 0 and shape[0] > 1:
                            if isinstance(indices, int):
                                item_result[k] = v[indices]
                            elif isinstance(indices, list) and len(indices) == 1:
                                item_result[k] = v[indices[0]]
                            elif isinstance(indices, slice):
                                item_result[k] = v[indices]
                            else:
                                sliced_items = [v[i] for i in indices]
                                item_result[k] = self._maybe_concatenate_brain_data(
                                    sliced_items
                                )
                        else:
                            item_result[k] = v
                    else:
                        item_result[k] = v
                result.append(item_result)
            return result
        else:
            # Direct Brain_Data object or list of them
            if not isinstance(data, list):
                data = [data]

            result = []
            for brain_data in data:
                if hasattr(brain_data, "__getitem__") and hasattr(brain_data, "shape"):
                    shape = brain_data.shape()
                    if len(shape) > 0 and shape[0] > 1:
                        if isinstance(indices, int):
                            result.append(brain_data[indices])
                        elif isinstance(indices, list) and len(indices) == 1:
                            result.append(brain_data[indices[0]])
                        elif isinstance(indices, slice):
                            result.append(brain_data[indices])
                        else:
                            sliced_items = [brain_data[i] for i in indices]
                            result.append(
                                self._maybe_concatenate_brain_data(sliced_items)
                            )
                    else:
                        result.append(brain_data)
                else:
                    result.append(brain_data)

            if len(result) == 1:
                return result[0]
            return self._maybe_concatenate_brain_data(result)

    def _format_output(
        self,
        data,
        is_single_position,
        is_single_key,
        is_single_label,
        actual_key=None,
        label=None,
    ):
        """Format the output based on intelligent flattening rules.

        Rules:
        - If exactly 2 out of 3 dimensions are singleton, return Brain_Data
        - Otherwise return Brain_Collection
        """
        singleton_count = sum([is_single_position, is_single_key, is_single_label])

        # If all 3 dimensions are singleton, or 2 out of 3 are singleton, return Brain_Data
        if singleton_count >= 2:
            # Intelligently flatten to Brain_Data
            if isinstance(data, list):
                # Concatenate all Brain_Data objects
                return self._maybe_concatenate_brain_data(data)
            elif isinstance(data, dict):
                # Single dict with multiple keys - can't flatten
                # This shouldn't happen with singleton_count >= 2
                pass
            else:
                # Already a Brain_Data
                return data

        # Return as Brain_Collection
        # Special case: when we have multiple positions but single key,
        # we get a list of Brain_Data objects that need to be wrapped
        if isinstance(data, list) and not is_single_position and is_single_key:
            # We have a list of Brain_Data objects from different positions
            # Need to wrap each in a dict with the appropriate key
            wrapped_data = []
            for item in data:
                if isinstance(item, dict):
                    wrapped_data.append(item)
                else:
                    # It's a Brain_Data object, wrap with the actual key
                    if actual_key:
                        wrapped_data.append({actual_key: item})
                    else:
                        wrapped_data.append({"result": item})

            collection = Brain_Collection(wrapped_data)
            if hasattr(self, "_labels"):
                collection._labels = self._labels
            return collection

        # General case
        if not isinstance(data, list):
            if isinstance(data, dict):
                data = [data]
            else:
                # Single Brain_Data, need to wrap in dict
                if actual_key:
                    data = [{actual_key: data}]
                else:
                    data = [{"result": data}]

        # If data contains Brain_Data objects directly, wrap them
        wrapped_data = []
        for item in data:
            if isinstance(item, dict):
                wrapped_data.append(item)
            else:
                # It's a Brain_Data object, need to wrap it
                if actual_key:
                    wrapped_data.append({actual_key: item})
                else:
                    wrapped_data.append({"result": item})

        collection = Brain_Collection(wrapped_data)
        # If we sliced by specific labels, update the collection's labels
        if hasattr(self, "_labels") and self._labels is not None and label is not None:
            # Check if we sliced labels
            if is_single_label and isinstance(label, str):
                collection._labels = [label]
            elif is_single_label and isinstance(label, int):
                collection._labels = [self._labels[label]]
            elif isinstance(label, list):
                # Get the label names for the indices
                label_names = []
                for l in label:
                    if isinstance(l, str):
                        label_names.append(l)
                    elif isinstance(l, int):
                        label_names.append(self._labels[l])
                collection._labels = label_names
            elif isinstance(label, slice) and label != slice(None):
                # Handle slice case
                collection._labels = self._labels[label]
            else:
                # Keep all labels
                collection._labels = self._labels
        elif hasattr(self, "_labels"):
            collection._labels = self._labels
        return collection

    def _maybe_concatenate_brain_data(self, result):
        """Check if result is a list of Brain_Data objects and concatenate if so.

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
        if not all(isinstance(item, Brain_Data) for item in result):
            return result

        # Always try to concatenate Brain_Data objects
        try:
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
    def labels(self):
        """Return the labels for this collection."""
        return self._labels

    def set_labels(self, labels, for_keys=None):
        """Set labels for this collection with validation.

        Args:
            labels: List of label names
            for_keys: None (validate all keys), string (single key), or list of strings (multiple keys)

        Raises:
            ValueError: If labels don't match the shape of specified Brain_Data objects
        """
        if not isinstance(labels, list):
            raise TypeError("Labels must be a list")

        # Determine which keys to validate
        if for_keys is None:
            keys_to_check = self._keys
        elif isinstance(for_keys, str):
            keys_to_check = [for_keys]
        elif isinstance(for_keys, list):
            keys_to_check = for_keys
        else:
            raise TypeError("for_keys must be None, str, or list of str")

        # Validate keys exist
        for key in keys_to_check:
            if key not in self._keys:
                raise ValueError(f"Key '{key}' not found in collection")

        # Validate labels match Brain_Data shapes
        for item in self.data:
            for key in keys_to_check:
                brain_data = item[key]
                if hasattr(brain_data, "shape"):
                    shape = brain_data.shape()
                    expected_len = shape[0] if len(shape) > 0 else 1
                    if len(labels) != expected_len:
                        raise ValueError(
                            f"Labels length {len(labels)} doesn't match "
                            f"Brain_Data shape {shape} for key '{key}'"
                        )

        self._labels = labels

    @property
    def shape(self):
        """Return shape as (n_items, n_keys)."""
        return (len(self.data), len(self._keys))

    def get(self, key, default=None):
        """Get values for a key with a default if key doesn't exist."""
        try:
            return self[key]  # Use __getitem__ with the key
        except KeyError:
            return default

    def __getattr__(self, name):
        """Enable attribute-style access to keys (e.g., collection.beta)."""
        if name in self._keys:
            return self[name]  # Use __getitem__ with the key
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )
