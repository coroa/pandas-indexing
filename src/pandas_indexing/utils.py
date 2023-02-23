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
