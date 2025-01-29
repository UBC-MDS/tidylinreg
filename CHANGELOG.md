# CHANGELOG


## v1.1.2 (2025-01-29)

### Bug Fixes

- (milestone 2 Feedback) add docstring for fixture function
  ([`d34cd08`](https://github.com/UBC-MDS/tidylinreg/commit/d34cd08e2083106a5b8dccc708444d3fb964bf0c))

- Adding docstrings for fit and predict test functions (Feedback from Milestone 2)
  ([`318ddf1`](https://github.com/UBC-MDS/tidylinreg/commit/318ddf1ae0b3a406012ea001676c90142ebfa2ac))

### Testing

- Added docstrings for tests in `get_std_error`
  ([`e91e03c`](https://github.com/UBC-MDS/tidylinreg/commit/e91e03c6528bf05e5e6677d410dbd6cc0079a903))

- Added test descriptions for `get_ci`
  ([`8bbec86`](https://github.com/UBC-MDS/tidylinreg/commit/8bbec860666db76108fde2381ab307bb9d5bcbc1))


## v1.1.1 (2025-01-28)

### Bug Fixes

- Changed `GITHUB_TOKEN` to `RELEASE_TOKEN` in `ci-cd.yml`
  ([`020e5ac`](https://github.com/UBC-MDS/tidylinreg/commit/020e5acf8037cc7a0081fbbb8fe5ca6950315e7d))

- Updated ci/cd workflow to run only on push/PR to `main`
  ([`c87b370`](https://github.com/UBC-MDS/tidylinreg/commit/c87b370cadf25597eb66215a644feb66b7ef9995))


## v1.1.0 (2025-01-25)

### Bug Fixes

- Added missing test for `get_ci` for non-numeric alpha
  ([`2f2803c`](https://github.com/UBC-MDS/tidylinreg/commit/2f2803ccf2c457badf54bf68b51f596ddae3904e))

- Capitalized Python in Intro
  ([`89fab61`](https://github.com/UBC-MDS/tidylinreg/commit/89fab617279c4df9057e7a980c4a405cff50e618))

- Fixed badge rendering issue
  ([`3f71416`](https://github.com/UBC-MDS/tidylinreg/commit/3f7141621ba9fa90d05e226379d5ad09d7a7be25))

- Fixed error in badge hyperlinks
  ([`750edf7`](https://github.com/UBC-MDS/tidylinreg/commit/750edf781161cada120f3a21a9e291bad41303b6))

- Markdown rendering
  ([`516d3ab`](https://github.com/UBC-MDS/tidylinreg/commit/516d3abdb7ac4cb6f6657d58dc76b2444f127edd))

- Modified test for `get_test_statistic` since it no longer returns a value
  ([`11db3b4`](https://github.com/UBC-MDS/tidylinreg/commit/11db3b4141590f1b9eaa21476d86e742fdd9a0bd))

- Raise TypeError in `summary` method for non-numeric `alpha` arg
  ([`8ea4e3d`](https://github.com/UBC-MDS/tidylinreg/commit/8ea4e3d2f8148053b2f9c4dbdfd3058c51f2b299))

- Rendering issues
  ([`098499a`](https://github.com/UBC-MDS/tidylinreg/commit/098499a18c74b960fd8852577e5858e2e6420bc9))

- Replaced `**kwargs` in `summary` method with `ci` and `alpha` arguments
  ([`f44e9b4`](https://github.com/UBC-MDS/tidylinreg/commit/f44e9b4913b0e9758ce44b67535697951a07aad5))

- Update readthedocs badge hyperlink to latest version
  ([`ebffeb1`](https://github.com/UBC-MDS/tidylinreg/commit/ebffeb132699e45d6e2ffefb788d7a29d587699a))

### Documentation

- Added additional documentation and ecosystem context
  ([`8f8d607`](https://github.com/UBC-MDS/tidylinreg/commit/8f8d60770f54e6e9a29c1dd0cf3b2349bf79c1d0))

- Added badge for readthedocs
  ([`84873ac`](https://github.com/UBC-MDS/tidylinreg/commit/84873acdaeb0f2deae8e30eb4cad8631879f18c1))

- Added more references and further elaboration on functions
  ([`38c010b`](https://github.com/UBC-MDS/tidylinreg/commit/38c010bfd6e6efac889fb5cdb1901baf3ca430dd))

- Added pytest tutorial
  ([`d3cf086`](https://github.com/UBC-MDS/tidylinreg/commit/d3cf086eeb51b0070384a406f294edb64b846162))

- Completed docstrings for `summary`
  ([`b4597bb`](https://github.com/UBC-MDS/tidylinreg/commit/b4597bbe891099736375447cee62b03eabe18e97))

- Completed docstrings for all methods except `summary`
  ([`f387197`](https://github.com/UBC-MDS/tidylinreg/commit/f387197104d9bbf521ece1805fc614b79552b8a5))

- Made changes to docstrings in response to comments
  ([`f49f94b`](https://github.com/UBC-MDS/tidylinreg/commit/f49f94b74f76a4227085ba1dd61cdd1a90258493))

- Organize TOC into Vignettes and Developer Notes
  ([`5391698`](https://github.com/UBC-MDS/tidylinreg/commit/5391698f72a7bb7e5cb49b35ad201bd2bb0ccb83))

- Update Contributors to list
  ([`877f610`](https://github.com/UBC-MDS/tidylinreg/commit/877f6106cd1f8897f194680db33f3a6a1b50921d))

### Features

- Added ci-cd badge
  ([`3345f18`](https://github.com/UBC-MDS/tidylinreg/commit/3345f18027a2a8060dc2a8ca4808472c41fc87bb))

- Added example usage for predict method
  ([`24fece2`](https://github.com/UBC-MDS/tidylinreg/commit/24fece2e70fa197f4e38585e632a510746e16dec))

- Added function example usage
  ([`52ed599`](https://github.com/UBC-MDS/tidylinreg/commit/52ed59948cdad8da7aba69f93710df83e373d313))


## v1.0.0 (2025-01-18)

### Build System

- Add numpy as a dependency
  ([`5293b0d`](https://github.com/UBC-MDS/tidylinreg/commit/5293b0df82d3a834bcab2ed649b37c57e4cc7bfe))

- Add pytest and pytest-cov as dev dependencies
  ([`61bdbf9`](https://github.com/UBC-MDS/tidylinreg/commit/61bdbf95135ff99be35b8ccfe73edd41ac9efe44))

- Added `statsmodels` as dependency for development
  ([`ebd7403`](https://github.com/UBC-MDS/tidylinreg/commit/ebd7403b6b7fff1a13a69986a58edbeb3a14ac69))

- Added dev dependencies for docs
  ([`c749a13`](https://github.com/UBC-MDS/tidylinreg/commit/c749a13ed52d4514f02eef1599d2bd77b74a459c))

- Added scipy as a dependency
  ([`968611d`](https://github.com/UBC-MDS/tidylinreg/commit/968611d321a8ed10084a974b62d1460f538f66cd))

- Remove upper bound on dependency versions
  ([`77eae1f`](https://github.com/UBC-MDS/tidylinreg/commit/77eae1f9e3f81647683e65adb3c9824c68555cc7))
