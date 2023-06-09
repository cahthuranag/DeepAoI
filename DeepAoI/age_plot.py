def plot_av_age_simulation(num_nodes, active_prob, n, k, P_range, num_iterations):
    from maincom import main_average_av_age
    from uncoded_age import uncoded_average_av_age
    import matplotlib.pyplot as plt

    av_age_simulations_ls = []
    av_age_simulations_uncoded_ls = []

    for P in P_range:
        av_age_sim_deep = main_average_av_age(num_nodes, active_prob, n, k, P, num_iterations)
        av_age_sim_uncoded = uncoded_average_av_age(num_nodes, active_prob, n, k, P, num_iterations)
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

plot_av_age_simulation(2, 0.5, 2, 2, np.concatenate([np.arange(0.05, 0.1, 0.01),np.arange(0.1, 1, 0.1), np.arange(1, 10, 1)]),10)

