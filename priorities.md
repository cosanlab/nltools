## Top priorities in order
1. Refactor library:
  - get library to stable, deployable state with all tests passing
  - refactor for better functional core, imperative shell design 
  - refactor to use new `nilearn` API features that replace old bespoke functionality
  - refactor to improve test coverage and prune tests that are implicitly covered by `nilearn` and other libraries we depend on
2. Refactor documentation:
  - Switch entirely from old sphinx & sphinx-gallery to jupyter-book
  - Port over all tutorials and make sure they are pedagogically useful and execute without errors (serves as 2nd layer of "workflow" testing)
3. Add new features to support working with collections of `Brain_Data` and `Model`(s) more generally
  - Should build upon existing classes or propose refactorings to make them amenable
  - Should be developed with easy-of-usability in mind as the top design property
  - Goal is to make it much easier to do common multi-step workflows that would otherwise require writing multiple lines of `nltools` and/or `nilearn` code, *without* hiding or abstracting away fine-grained control (i.e. "glass-box" philosophy)
  