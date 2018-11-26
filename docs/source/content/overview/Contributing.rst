Contributing
============

Contributions to modAL are very much welcome! If you would like to help in general, visit the Issues page, where you'll find bugs to be fixed, features to be implemented. If you have a concrete feature in mind, you can proceed as follows.

1. Open a new issue. This helps us to discuss your idea and makes sure that you are not working in parallel with other contributors.

2. Fork the modAL repository and clone your fork to your local machine:

.. code:: bash
    
    $ git clone git@github.com:username/modAL.git


3. Create a feature branch for the changes from the dev branch:

.. code:: bash

    $ git checkout -b new-feature dev


Make sure that you create your branch from ``dev``.

4. After you have finished implementing the feature, make sure that all the tests pass. The tests can be run as

.. code:: bash
    
    $ python3 path-to-modAL-repo/tests/core_tests.py

5. Commit and push the changes.

.. code:: bash
    
    $ git add modified_files
    $ git commit -m 'commit message explaning the changes briefly'
    $ git push origin new-feature3

6. Create a pull request from your fork **to the dev branch**. After the code is reviewed and possible issues are cleared, the pull request is merged to ``dev``.
