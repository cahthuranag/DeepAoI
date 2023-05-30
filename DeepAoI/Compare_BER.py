def compare_ber(n, k, snr_range):
    import numpy as np
    import matplotlib.pyplot as plt
    from deepencoder import deepencoder
    from deepencoder import ber_bpsk
    ber_deep_encoder = []
    ber_uncoded_bpsk = []
    
    for snr in snr_range:
        # Calculate the BER of the deep encoder
        ber_deep_encoder.append(deepencoder(n, k, snr))
        
        # Calculate the BER of uncoded BPSK
        ber_uncoded_bpsk.append(ber_bpsk(snr))
    
    # Plot the results
    plt.figure()
    plt.plot(snr_range, ber_deep_encoder, label='Deep Encoder')
    plt.plot(snr_range, ber_uncoded_bpsk, label='Uncoded BPSK')
    plt.xlabel('SNR (linear)')
    plt.ylabel('Block Error Rate (BER)')
    plt.title('BER Comparison: Deep Encoder vs. Uncoded BPSK')
    plt.legend()
    plt.grid(True)
    plt.show()
import numpy as np
snr_range = np.arange(1, 10, 1)  
compare_ber(8, 4, snr_range)
