Contributing
============

.. start-here

There are several strategies on how to contribute to a project on github. Here,
I explain the one I use for all the project I am participating. You can use this
same strategy to contribute to this template or to suggest contributions to your
project.

Fork this repository
--------------------

`Fork this repository before contributing`_. It is a better practice, possibly
even enforced, that only pull request from forks are accepted. In my opinion
enforcing forks creates a cleaner representation of the `contributions to the
project`_.

Clone the main repository
~~~~~~~~~~~~~~~~~~~~~~~~~

Next, clone the main repository to your local machine:

::

    git clone https://github.com/coroa/pandas-indexing.git
    cd pandas-indexing

Add your fork as an upstream repository:

::

    git remote add myfork git://github.com/YOUR-USERNAME/pandas-indexing.git
    git fetch myfork

Install for developers
----------------------

Create a dedicated Python environment where to develop the project.

If you are using :code:`pip` follow the official instructions on `Installing
packages using pip and virtual environments`_, most likely what you want is:

::

    python -m venv venv
    source venv/bin/activate

If you are using `Anaconda`_ go for:

::

    mamba create --name pandas-indexing python
    mamba activate pandas-indexing

Where :code:`pandas-indexing` is the name you wish to give to the environment
dedicated to this project.

Either under *pip* or *mamba*, install the package in :code:`develop` mode.
Install also :ref:`tox<Uniformed Tests with tox>`.

::

    pip install -e ".[docs,test,lint]"
    pip install tox

This configuration, together with the use of the ``src`` folder layer,
guarantees that you will always run the code after installation. Also, thanks to
the ``editable`` flag, any changes in the code will be automatically reflected in
the installed version.

Make a new branch
-----------------

From the ``main`` branch create a new branch where to develop the new code.

::

    git switch main
    git switch -c new_branch


**Note** the ``main`` branch is from the main repository.

Develop the feature and keep regular pushes to your fork with comprehensible
commit messages.

::

    git status
    git add (the files you want)
    git commit (add a nice commit message)
    git push myfork new_branch

While you are developing, you can execute ``tox`` as needed to run your unit
tests or inspect lint, or other integration tests. See the last section of this
page.

Update your branch
~~~~~~~~~~~~~~~~~~

It is common that you need to keep your branch update to the latest version in
the ``main`` branch. For that:

::

    git switch main  # return to the main branch
    git pull  # retrieve the latest source from the main repository
    git switch new_branch  # return to your devel branch
    git merge --no-ff main  # merge the new code to your branch

At this point you may need to solve merge conflicts if they exist. If you don't
know how to do this, I suggest you start by reading the `official docs
<https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/addressing-merge-conflicts/resolving-a-merge-conflict-on-github>`_

You can push to your fork now if you wish:

::

    git push myfork new_branch

And, continue doing your developments are previously discussed.

Update CHANGELOG
~~~~~~~~~~~~~~~~

Update the changelog file under :code:`CHANGELOG.rst` with an explanatory bullet
list of your contribution. Add that list right after the main title and before
the last version subtitle:

::

    Changelog
    =========

    * here goes my new additions
    * explain them shortly and well

    vX.X.X (1900-01-01)
    -------------------

Also add your name to the authors list at :code:`docs/AUTHORS.rst`.

Pull Request
~~~~~~~~~~~~

Once you finished, you can create a pull request to the main
repository, and engage with the community.

**Before submitting a Pull Request, verify your development branch passes all
tests as** :ref:`described below <Uniformed Tests with tox>` **. If you are
developing new code you should also implement new test cases.**


Uniformed Tests with tox
------------------------

Thanks to `Tox`_ we can have a unified testing platform that runs all tests in
controlled environments and that is reproducible for all developers. In other
words, it is a way to welcome (*force*) all developers to follow the same rules.

The ``tox`` testing setup is defined in a configuration file, the
`tox.ini`_, which contains all the operations that are performed during the test
phase. Therefore, to run the unified test suite, developers just need to execute
``tox``, provided `tox is installed`_ in the Python environment in use.

::

    pip install tox
    # or
    mamba install tox -c conda-forge


Before creating a Pull Request from your branch, certify that all the tests pass
correctly by running:

::

    tox

These are exactly the same tests that will be performed online in the Github
Actions.

Also, you can run individual testing environments if you wish to test only specific
functionalities, for example:

::

    tox -e lint  # code style
    tox -e build  # packaging
    tox -e docs  # only builds the documentation
    tox -e test  # runs unit tests


.. _tox.ini: https://github.com/coroa/pandas-indexing/blob/latest/tox.ini
.. _Tox: https://tox.readthedocs.io/en/latest/
.. _tox is installed: https://tox.readthedocs.io/en/latest/install.html
.. _MANIFEST.in: https://github.com/coroa/pandas-indexing/blob/main/MANIFEST.in
.. _Fork this repository before contributing: https://github.com/coroa/pandas-indexing/network/members
.. _up to date with the upstream: https://gist.github.com/CristinaSolana/1885435
.. _contributions to the project: https://github.com/coroa/pandas-indexing/network
.. _Gitflow Workflow: https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow
.. _Pull Request: https://github.com/coroa/pandas-indexing/pulls
.. _PULLREQUEST.rst: https://github.com/coroa/pandas-indexing/blob/main/docs/PULLREQUEST.rst
.. _1: https://git-scm.com/docs/git-merge#Documentation/git-merge.txt---no-ff
.. _2: https://stackoverflow.com/questions/9069061/what-is-the-difference-between-git-merge-and-git-merge-no-ff
.. _Installing packages using pip and virtual environments: https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment
.. _Anaconda: https://www.anaconda.com/
