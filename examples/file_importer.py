"""Example for loading data into an optimus File.

The provided json file should look something like the following:

{
  "some_other_key": {
    "numpy_file": "/path/to/a/different/file.npy",
    "label": [
      1,
      2,
      3
    ]
  },
  "im_a_key": {
    "numpy_file": "/path/to/a/file.npy",
    "label": "saxamaphone"
  }
}

"""

import argparse
import json
import numpy as np
import optimus
import time


def item_to_entity(item, dtype=np.float32):
    """Create an entity from the given item.

    This function exists primarily as an example, and is quite boring. However,
    it expects that each item dictionary has two keys:
        - numpy_file: str
            A valid numpy file on disk
        - label: obj
            This can be any numpy-able datatype, such as scalars, lists,
            strings, or numpy arrays. Dictionaries and None are unsupported.

    Parameters
    ----------
    item: dict
        Contains values for 'numpy_file' and 'label'.
    dtype: type
        Data type to load the requested numpy file.
    """
    data = np.load(item['numpy_file'])
    return optimus.Entity(data=data.astype(dtype), label=item['label'])


def data_to_file(data, file_handle, item_parser, dtype=np.float32):
    """Load a label dictionary into an optimus file.

    Parameters
    ----------
    data: dict of dicts
        A collection of data to load, where the keys of ``data`` will become
        the keys in the file, and the corresponding values are sufficient
        information to load data into an Entity.
    file_handle: optimus.File
        Open for writing data.
    item_parser: function
        Function that consumes a dictionary of information and returns an
        optimus.Entity. Must take ``dtype`` as an argument.
    """
    total_count = len(data)
    for idx, key in enumerate(data.iterkeys()):
        file_handle.add(key, item_parser(data[key], dtype))
        print "[%s] %12d / %12d: %s" % (time.asctime(), idx, total_count, key)


def main(args):
    """Main routine for importing data."""
    fhandle = optimus.File(args.output_file)
    data = json.load(open(args.json_file))

    # More interesting situations will require a different item_parser!!
    data_to_file(data, fhandle, item_to_entity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute CQT representations for a "
                    "collection of audio files")
    parser.add_argument("json_file",
                        metavar="json_file", type=str,
                        help="JSON file to load into an optimus file.")
    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="Filename for the output.")

    main(parser.parse_args())
