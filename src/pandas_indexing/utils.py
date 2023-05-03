"""
Utils module.

Simple utility functions not of greater interest
"""


def shell_pattern_to_regex(s):
    """
    Escape characters with specific regexp use.
    """
    return (
        str(s)
        .replace("|", r"\|")
        .replace(".", r"\.")  # `.` has to be replaced before `*`
        .replace("**", "__starstar__")  # temporarily __starstar__
        .replace("*", r"[^|]*")
        .replace("__starstar__", r".*")
        .replace("+", r"\+")
        .replace("(", r"\(")
        .replace(")", r"\)")
        .replace("$", r"\$")
    )


def print_list(x, n):
    """
    Return a printable string of a list shortened to n characters.

    Copied from pyam.utils.print_list by Daniel Huppmann, licensed under Apache 2.0.

    https://github.com/IAMconsortium/pyam/blob/449b77cb1c625b455e3675801477f19e99b30e64/pyam/utils.py#L599-L638 .
    """
    # if list is empty, only write count
    if len(x) == 0:
        return "(0)"

    # write number of elements, subtract count added at end from line width
    x = [i if i != "" else "''" for i in map(str, x)]
    count = f" ({len(x)})"
    n -= len(count)

    # if not enough space to write first item, write shortest sensible line
    if len(x[0]) > n - 5:
        return "..." + count

    # if only one item in list
    if len(x) == 1:
        return f"{x[0]} (1)"

    # add first item
    lst = f"{x[0]}, "
    n -= len(lst)

    # if possible, add last item before number of elements
    if len(x[-1]) + 4 > n:
        return lst + "..." + count
    else:
        count = f"{x[-1]}{count}"
        n -= len({x[-1]}) + 3

    # iterate over remaining entries until line is full
    for i in x[1:-1]:
        if len(i) + 6 <= n:
            lst += f"{i}, "
            n -= len(i) + 2
        else:
            lst += "... "
            break

    return lst + count
