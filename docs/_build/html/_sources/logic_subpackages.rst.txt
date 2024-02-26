Logic of Subpackages
--------------------

The helpers and utils subpackages are documented for transparency. We assign 
modules to one of the two packages as far as possible on the basis of the 
following criteria.

`helpers`

- Depend on other parts and/or external libraries
- Rather support specific parts of the package
- E.g., validation functions, data formatting classes, ...

  → Architectural snippets

`helpers.base`

Contains helpers that are used to build other helpers. They do not depend on 
other helpers.

`utils`

- Independent of other parts (no internal imports)
- Used generally
- E.g., string manipulations, math libraries, ...

  → Snippets to build bigger units

.. note::
    Note that we use `_` as prefix for modules and methods that are not 
    imported in the directory of the main file. Modules/methods starting with 
    `_` are excluded from this documentation.