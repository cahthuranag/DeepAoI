def plot_av_age_simulation(num_nodes, active_prob, n, k, P_range):
    from maincom import main
    from uncoded_age import uncoded_age
    import matplotlib.pyplot as plt

    av_age_simulations_ls = []
    av_age_simulations_uncoded_ls = []

    for P in P_range:
        av_age_sim_deep = main(num_nodes, active_prob, n, k, P)
        av_age_sim_uncoded = uncoded_age(num_nodes, active_prob, n, k, P)
        av_age_simulations_uncoded_ls.append(av_age_sim_uncoded)
        av_age_simulations_ls.append(av_age_sim_deep)

    plt.plot(P_range, av_age_simulations_ls, marker='o', label='Deep encoder')
    plt.plot(P_range, av_age_simulations_uncoded_ls, marker='o', label='Uncoded')
    plt.xlabel('Transmission Power (W))')
    plt.ylabel('Average Age of Information')
    plt.title('Average Age of Information vs Transmission Power')
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()

import numpy as np

plot_av_age_simulation(10, 0.5, 8, 4, [np.arange( 0.01, 0.01, 0.1), np.arange(0.1, 0.1, 1), np.arange(1, 5, 1)])

