def compare_ber(n, k, snr_range):
    """
    Compare the BER of the deep encoder with uncoded BPSK.
    Args:
        n: Block length of the code.
        k: Message length of the code.
        snr_range: Range of SNR values to test.
    Returns:
     fig: Comparison of BER for deep encoder and uncoded BPSK.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from deepencoder import deepencoder
    from deepencoder import ber_bpsk
    ber_deep_encoder = []
    ber_uncoded_bpsk = []
    
    for snr in snr_range:
        # Calculate the BER of the deep encoder
        ber_deep_encoder.append(deepencoder(n, k, snr)) # Calculate the BER of the deep encoder
        
        # Calculate the BER of uncoded BPSK
        ber_uncoded_bpsk.append(ber_bpsk(snr,n)) # Calculate the BER of uncoded BPSK
    
    # Plot the results
    plt.figure()
    plt.plot(snr_range, ber_deep_encoder, label='Deep Encoder') # Plot the BER of the deep encoder
    plt.plot(snr_range, ber_uncoded_bpsk, label='Uncoded BPSK') # Plot the BER of uncoded BPSK
    plt.xlabel('SNR (linear)')
    plt.ylabel('Block Error Rate (BER)')
    plt.title('BER Comparison: Deep Encoder vs. Uncoded BPSK')
    plt.legend()
    plt.grid(True)
    plt.show()

def compare_ber_list(n_list, k_list, snr_range):
    """
    Compare the BER of the deep encoder with uncoded BPSK for different n, k combinations.
    Args:
        n_list: List of block lengths of the codes.
        k_list: List of message lengths of the codes.
        snr_range: Range of SNR values to test.
    Returns:
        fig: Comparison of BER for deep encoder and uncoded BPSK.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from deepencoder import deepencoder
    from deepencoder import ber_bpsk
    
    plt.figure()
    
    for n, k in zip(n_list, k_list):
        ber_deep_encoder = []
        ber_uncoded_bpsk = []
        
        for snr in snr_range:
            # Calculate the BER of the deep encoder
            ber_deep_encoder.append(deepencoder(n, k, snr))
            
            # Calculate the BER of uncoded BPSK
            ber_uncoded_bpsk.append(ber_bpsk(snr,n))
        
        # Plot the results
        label = f'n={n}, k={k}'
        plt.plot(snr_range, ber_deep_encoder, label=f'Deep Encoder ({label})')
        plt.plot(snr_range, ber_uncoded_bpsk, label=f'Uncoded BPSK ({label})')
    
    plt.xlabel('SNR (linear)')
    plt.ylabel('Block Error Rate (BER)')
    plt.title('BER Comparison: Deep Encoder vs. Uncoded BPSK')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.xscale('log')
    plt.show()




import numpy as np
snr_range = np.arange(1, 10, 1)  
#compare_ber(2, 2, snr_range)

n_list = [2,8]  # List of block lengths
k_list = [2,4]  # List of message lengths
snr_range = np.concatenate([np.arange(0.1, 1, 0.2), np.arange(1, 10, 0.5)]) # Range of SNR values

compare_ber_list(n_list, k_list, snr_range)
