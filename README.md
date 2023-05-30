[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/cahthuranag/DeepAoI/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# DeepAoI
A Python 3.8 implementation of the System Model estimates the average AoI (AAoI) in a deep learning-aided multi-hop wireless communication system over the AWGN channels.



The DeepAoI contains several functions that can be used to study the Age of Information (AoI) in a multi-hop wireless network. These functions include:
- main: This function takes input parameters such as the number of nodes, active probability, block size, message size, and transmission power. It simulates the communication system by generating arrival and departure timestamps for events, considering factors like noise power, distance, and signal-to-noise ratio (SNR). The function also utilizes external functions from modules av_age, snr, and deepencoder. Finally, it returns the simulated AAoI
-  deepencoder : this  function is a deep learning-based encoder designed for an AWGN channel. It takes the number of bits in a block (n), the number of bits in a message (k), and the SNR as inputs. This function builds and trains a deep neural network model using Keras and TensorFlow. The model consists of an encoder and a decoder, which are trained to minimize the categorical cross-entropy loss. It encodes messages, adds noise based on the SNR, and decodes the noisy signals to compute the Block Error Rate (BER). The function provides a BER value as the output, indicating the accuracy of the encoding and decoding process in the presence of AWGN.
-  average_age_of_information_fn: This function calculates the average age of information based on destination times, generation times, and arrival rate. It uses a time step to generate a time array and calculates the age at each time step by subtracting the offset. The average age is then computed by integrating the age versus time curve.
  
-   compare_ber: This  function is designed to compare the performance of a deep encoder and uncoded BPS scheme in terms of Block Error Rate (BER).


## Result

This figure illustrates BER vs. SNR for both the deep learning-based encoder and the uncoded BPSK BER.

![BER.](https://github.com/cahthuranag/DeepAoI/blob/main/image/Figure_1.png)

This figure illustrates AAoI vs. Transmission power for both learning-based encoder and the uncoded BPSK 

![AAoI](https://github.com/cahthuranag/DeepAoI/blob/main/image/Figure_2.png)
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




## License

[MIT License](LICENSE)

