"""
Configuration file for then Sphinx documentation builder.
"""


# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
from importlib.metadata import version


# -- Project information --------------------------------------------------------------

source_suffix = ".rst"
master_doc = "index"
project = "pandas-indexing"
year = "2020"
author = ", ".join(["Jonas Hörsch"])
copyright = f"{year}, {author}"

# Retrieve package version from installed metadata
release = version("pandas-indexing")
version = ".".join(release.split(".")[:3])


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    # "sphinx.ext.ifconfig",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    # "sphinx.ext.todo",
    # "sphinx.ext.viewcode",
    "nbsphinx",
]

if os.getenv("SPELLCHECK"):
    extensions += ("sphinxcontrib.spelling",)
    spelling_show_suggestions = True
    spelling_lang = "en_US"
    # https://sphinxcontrib-spelling.readthedocs.io/en/latest/customize.html
    spelling_word_list_filename = ["spelling_wordlist.txt"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["."]


extlinks = {
    "issue": (
        "https://github.com/coroa/pandas-indexing/issues/%s",
        "GH%s",
    ),
    "pull": ("https://github.com/coroa/pandas-indexing/pull/%s", "PR%s"),
}

# codecov io closes connection if host is accessed too repetitively.
# codecov links are ignored here for the same reason there's a sleep
# in the .travis.yml file
# see https://github.com/codecov/codecov-python/issues/158
linkcheck_ignore = [
    "https://codecov.io/gh/coroa/pandas-indexing/*",
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["html"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_path = ["_static"]

html_context = {
    "display_github": False,
    "github_user": "coroa",
    "github_repo": "pandas-indexing",
    "github_version": "main",
    "conf_py_path": "/docs/source",
}


# -- Extension configuration -------------------------------------------------

# -- Options for coverage extension ------------------------------------------
coverage_write_headline = False  # do not write headlines.

# -- Options for autodoc extension -------------------------------------------

# Do not add module names in the doc to hide the internal package structure of SeisBench
add_module_names = False
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "special-members": False,
    "inherited-members": True,
    "show-inheritance": True,
}

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "python": ("https://docs.python.org/3", None),
    "pyam": ("https://pyam-iamc.readthedocs.io/en/latest", None),
    "scmdata": ("https://scmdata.readthedocs.io/en/latest", None),
    # "pint": ("https://pint.readthedocs.io/en/latest", None), # no full API doc here, unfortunately
}

# -- Options for napoleon extension ------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
set_type_checking_flag = False

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True
