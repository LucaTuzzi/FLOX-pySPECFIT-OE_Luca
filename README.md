# Python conversion

This is a python "raw" conversion of the matlab code provided.
It tries to emulate the same structure (with some slight differences).

This conversion is able to reproduce the MATLAB code with machine precision except in the
uncertainties computation since the random generation in python follows a different algorithm than MATLAB's.

This is not an industrialized tool, but will serve has basis for development.

## Execution procedure

### Prerequisites

Python 3.10+ (recommended: 3.12.10, used during development)
'pip' installed (<https://pypi.org/project/pip/>)


### (optional) create a virtual environment 
```sh
python -m venv /path/to/new/virtual/environment
```

### Activate the environment on Linux/macOS with:
```sh
source /path/to/new/virtual/environmenbin/activate
```
On Windows:
\path\to\new\virtual\environment\Scripts\activate


### install required python libraries
```sh
pip install -r requirements.txt
```
### Optional dependencies
Parallel execution support is optional and not required for the standard workflow.
To enable parallel processing features, install additional packages:
```sh
pip install -r requirements-optional.txt
```

### Run the tool with:
```sh
python3 CALL_FLOX_processing.py
```

The output (and Log files) is written in ./data_path/test_tds (automatically created if it does not exist)

The project supports 2 input formats, NetCDF and CSV.

Test datasets are currently organized into three separate folders: NetCDF (.nc), CSV with uncertainty, and CSV without uncertainty. In future versions, these will be unified into a single dataset folder containing both formats.
Note: these folders are excluded from version control via `.gitignore` and are therefore not uploaded to the GitHub repository.