"""
Config file for Sphinx-docs.
"""
import os
import sys
from importlib.metadata import version
from unittest import mock

import sphinx_py3doc_enhanced_theme


mock_modules = [
    "matplotlib",
]

for modulename in mock_modules:
    sys.modules[modulename] = mock.Mock()

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.ifconfig",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
]

todo_include_todos = True

exclude_patterns = [
    "nonlisted/*.rst",
]

if os.getenv("SPELLCHECK"):
    extensions += ("sphinxcontrib.spelling",)
    spelling_show_suggestions = True
    spelling_lang = "en_US"
    # https://sphinxcontrib-spelling.readthedocs.io/en/latest/customize.html
    spelling_word_list_filename = ["spelling_wordlist.txt"]

source_suffix = ".rst"
master_doc = "index"
project = "pandas-indexing"
year = "2020"
author = "Jonas Hörsch"
copyright = f"{year}, {author}"

# Retrieve package version from installed metadata
release = version("pandas-indexing")
version = ".".join(release.split(".")[:2])

pygments_style = "trac"
templates_path = ["."]
extlinks = {
    "issue": ("https://github.com/coroa/pandas-indexing/issues/%s", "#"),  # noqa: E501
    "pr": ("https://github.com/coroa/pandas-indexing/pull/%s", "PR #"),  # noqa: E501
}

# codecov io closes connection if host is accessed too repetitively.
# codecov links are ignored here for the same reason there's a sleep
# in the .travis.yml file
# see https://github.com/codecov/codecov-python/issues/158
linkcheck_ignore = [
    "https://codecov.io/gh/coroa/pandas-indexing/*",
]

html_theme = "sphinx_py3doc_enhanced_theme"
html_theme_path = [sphinx_py3doc_enhanced_theme.get_html_theme_path()]
html_theme_options = {
    "githuburl": "https://github.com/coroa/pandas-indexing",
}

html_use_smartypants = True
html_last_updated_fmt = "%b %d, %Y"
html_split_index = False
html_sidebars = {
    "**": ["searchbox.html", "globaltoc.html", "sourcelink.html"],
}
html_short_title = f"{project}-{version}"

napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False
