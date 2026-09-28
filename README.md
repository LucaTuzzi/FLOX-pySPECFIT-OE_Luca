## 1) Python conversion

This is a python "raw" conversion of the matlab code provided.
It tries to emulate the same structure (with some slight differences).

This conversion is able to reproduce the MATLAB code with machine precision except in the
uncertainties computation since the random generation in python follows a different algorithm than MATLAB's.

This is not an industrialized tool, but will serve has basis for development.

## 2) Execution procedure

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



## 3) Retrieval algorithm (SIF core module)
The following diagram shows the internal structure of the SIF retrieval engine implemented in `sif_retrieval_IPF_v2_1_modified.py`.

**CALL_FLOX_processing()**                    (CALL_FLOX_processing.py)
│
└── FLOX_processing_master()              (FLOX_processing_master.py)
    │
    ├── FLOX_processing()                 (FLOX_processing.py)
    │   │
    │   ├── process_spectrum()            (FLOX_processing.py)
    │   │   │
    │   │   └── _l2b_regularized_cost_function_optimization()
    │   │           (sif_retrieval_IPF_v2_1_modified.py)
    │   │           │
    │   │           └──Retrieval core*
    │   │
    │   └── sif_parms_flox()              (FLOX_processing.py)
    │
    ├── load_inverse_covariance()         (FLOX_functions.py)
    ├── write_csv_with_headers()          (FLOX_functions.py)
    ├── write_results_to_netcdf()         (FLOX_functions.py)
    └── FLOX_mask()                       (FLOX_mask.py)

Retrieval core*   (sif_retrieval_IPF_v2_1_modified.py)
│
├── l2b_forward_model()
│   ├── _compute_reflectance()
│   ├── _compute_fluorescence()
│   └── _sif_forward_model()
│
├──  Spectral modeling
│   ├── _compute_asymmetric_super_gaussian()
│   └── _analytical_barycenter()
│
├── Regularization / optimization
│   ├── compute_LM_gain_ratio()
│   ├── LMgamma_update_Geodesic()
│   └── MC_SIF_uncertainty_estimation()
│
└── Jacobian block
    ├── _jacobian_calculation()
    └── _jacobian_calculation_analytical()
