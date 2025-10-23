# SARU-Net
A Shadow-Aware and Removal Unified Network for Remote Sensing Images with New Benchmarks
.
├── Dataset/ # Custom dataset definition (dataset.py)
├── utils/
│ ├── DBSCF_DCENet.py # Shadow Detection Network model definition
│ └── model.py # (Potentially other model definitions, not explicitly used in provided code)
├── boundary/
│ └── boundary_smooth.py # Boundary smoothing algorithm for shadow removal
├── bath_sr.py # Batch shadow removal script (used by the main function in bath_sr.py itself, or for specific batch processing)
├── dataset.py # Dataset class for loading images and masks
├── demo.py # Script for a full shadow detection and removal demonstration on a single image
├── test.py # Script for testing shadow detection performance
├── train.py # Script for training the shadow detection network
├── quantity.py # (Assumed) Script for quantitative evaluation of shadow removal
└── requirements.txt # Python dependencies
