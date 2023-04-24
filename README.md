
## Faux ~~<sub>**Net**</sub>~~

#### A Toolset for assisting the development of AI-enhanced Algorithmic Trading Systems

---

## Notes:
 - **very** much a work-in-progress, and frankly a mess
 - much of the code present in the repository is partially written, experimental, or just flat-out broken!
   - in such cases, the code will most likely not contain any notes to say that it is (incomplete, an old experiment, broken), so caution is strongly advised.
 - the code that does work as intended is often poorly documented
 - use at your own risk, and in so doing understand that neither I nor any other contributors (direct or indirect) to this repo are responsible in any way, for any effect or outcome brought about by the evaluation (be it by a Python interpreter, or any other means by which code might be interpreted) of any code found in this repository, now or at any time in the future. 
 **YOU (i.e. the "User") TAKE FULL RESPONSIBILITY** for any/all effects of evaluation of said code, regardless of the desirability of those effects; 
 
> You have been warned

---

## Features and Capabilities
 - Decent backtesting support, via `faux.backtesting.Backtest`
 - Fluent composition of parameter-grids for hyperparameter tuning, via `faux.PGrid`

## Usage Guide

  Basic Usage:
  ```bash
  # from project root
  python3.8 main.py <cmd> [options]
  ```

  Available Commands:
   - 
### Available Commands
 
  

## TODOs / Planned Features
 - Fluent composition of TS-processing procedures, via `faux.TSFnPipeline`
 - 
 - Automatic conversion of pinescript source code into an equivalent Python implementation. The usefulness may not be obvious, but it would make considerably more technical-indicators available within this framework without having to manually implement all of indicators not found in any of the major TA libraries for Python.
   - Possibly also partial support for conversion of EasyLanguage scripts to Python, though from my limited knowledge of the subject, this would probably not be useful in very many cases
 - ...
