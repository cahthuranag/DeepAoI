# DeepAoI
A Python 3.8 implementation of the System Model estimates the average AoI (AAoI) in a deep learning-aided multi-hop wireless communication system over the AWGN channels.

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

