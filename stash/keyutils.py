

def expand_hex(hexval, width):
    """Zero-pad a hexadecimal representation out to a given number of places.

    Example:

    Parameters
    ----------
    hexval : str
        Hexadecimal representation, produced by hex().
    width : int
        Number of hexadecimal places to expand.

    Returns
    -------
    padded_hexval : str
        Zero-extended hexadecimal representation.

    Note: An error is raised if width is less than the number of hexadecimal
    digits required to represent the number.
    """
    chars = hexval[2:]
    assert width >= len(chars), \
        "Received: %s. Width (%d) must be >= %d." % (hexval, width, len(chars))
    y = list('0x' + '0' * width)
    y[-len(chars):] = list(chars)
    return "".join(y)


def index_to_hexkey(index, depth):
    """Convert an integer to a hex-key representation.

    Example: index_to_hexkey(843, 2) -> '03/4b'

    Parameters
    ----------
    index : int
        Integer index representation.
    depth : int
        Number of levels in the key (number of slashes plus one).

    Returns
    -------
    key : str
        Slash-separated hex-key.
    """
    hx = expand_hex(hex(int(index)), depth * 2)
    tmp = ''.join(
        [''.join(d) for d in zip(hx[::2], hx[1::2], '/' * (len(hx) / 2))])
    return tmp[3:-1]


def uniform_hexgen(depth, width=256):
    """Generator to produce uniformly distributed hexkeys at a given depth.

    Deterministic and consistent, equivalent to a strided xrange() that yields
    strings like '04/1b/22' for depth=3, width=256.

    Parameters
    ----------
    depth : int
        Number of nodes in a single branch. See docstring in keyutil.py for
        more information.
    width : int
        Child nodes per parent. See docstring in keyutil.py for more
        information.

    Returns
    -------
    key : str
        Hexadecimal key path.
    """
    max_index = width ** depth
    index = 0
    for index in xrange(max_index):
        v = expand_hex(hex(index), depth * 2)
        hexval = "0x" + "".join([a + b for a, b in zip(v[-2:1:-2], v[:1:-2])])
        yield index_to_hexkey(int(hexval, 16), depth)
    raise ValueError("Unique keys exhausted.")
