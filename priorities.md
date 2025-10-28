## Key Question Responses
1. See the top-priorities below
2. We should never unneccsarily reproduce functionality natively available in `nilearn`. We should instead aim to make it more useable and automate common operations that increase usability.
3. Our current testing strategy is pretty haphazard so we should definitely be more systematic starting with making sure all the functionality we offered in the last release (version 0.5.1, see git history) is covered and tested. As we refactor, we should adapt our testing strategies accordingly.
4. Remind yourself in CLAUDE.md to always use claude-research/ to as your working knowledge base that you should use sub-agents to add to when more research is required
5. Ignore performance testing for now. But make sure that any refactors we make take performance into account
6. Ignore this for now
7. Our target is version `0.6.0` which will be a backwards incompatible release with breaking API changes in service of future-facing features and stability

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
  