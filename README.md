[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/cahthuranag/Agewire/blob/3000891c482e715b3006264a88dfcf4ed4aedc7c/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![GitHub Repo stars](https://img.shields.io/github/stars/cahthuranag/DeepAoI?style=social)
![GitHub last commit](https://img.shields.io/github/last-commit/cahthuranag/DeepAoI)
# DeepAoI
A Python 3.8 implementation of the System Model estimates the average AoI (AAoI) in a deep learning-aided multi-hop wireless communication system over the AWGN channels.



The DeepAoI contains several functions that can be used to study the Age of Information (AoI) in a multi-hop wireless network. These functions include:
**main: This function takes input parameters such as the number of nodes, active probability, block size, message size, and transmission power. It simulates the communication system by generating arrival and departure timestamps for events, considering factors like noise power, distance, and signal-to-noise ratio (SNR). The function also utilizes external functions from modules av_age, snr, and deepencoder. Finally, it returns the simulated AAoI


## Result


## Requirements

The implementation requires Python 3.8+ to run.
The following libraries are also required:

- `numpy`
- `matplotlib`
- `pandas`
- `argparse`
- `itertools`
- `math`
- `scipy`
- `random`
- `tensorflow`
- `keras`

## How to install

### From PyPI

```
pip install 
```

### From source/GitHub

Directly using pip:

```
pip install 
```

Or each step at a time:

```
git clone  https://github.com/cahthuranag/DeepAoI.git
cd 
pip install .
```

### Installing for development and/or improving the package

```
git clone hhttps://github.com/cahthuranag/DeepAoI.git
cd 
pip install -e .[dev]
```

This way, the package is installed in development mode. As a result, the pytest dependencies/plugins are also installed.

## Documentation

* [*DeepAoI* package documentation]()


## License

[MIT License](LICENSE)

