
def main(
        num_nodes: int,
        active_prob: float,
        n: int,
        k: int,
        P: float,
       ):
    import random
    import numpy as np

    from av_age import average_age_of_information_fn 
    from snr import snr_th 
    from deepencoder import deepencoder
    """
    Simulates a communication system and calculates the AAoI.

    Args:
        num_nodes (int): Number of nodes in the system
        active_prob (float): Probability that a node is active.
        n (int): Number of bits in a block.
        k (int): Number of bits in a message.
        P (float): Power of the nodes.

    Returns:
    simulation AAoI.
    """
    lambda1 = 1  # arrival for one transmission period
    # lambda1 = genlambda[j]
    num_events = 10000  # number of events
    inter_arrival_times = (1 / lambda1) * \
        (np.ones(num_events))  # inter arrival times
    arrival_timestamps = np.cumsum(inter_arrival_times)  # arrival timestamps
    N0 = 1 * (10**-11)  # noise power
    d1 = 700  # disatance between source nodes and destination
    snr1_th = snr_th(N0, d1, P)
    inter_service_times = (1 / lambda1) * \
        np.ones((num_events))  # inter service times
    # Generating departure timestamps for the node 1
    server_timestamps_1 = np.zeros(num_events)
    departure_timestamps_s = np.zeros(num_events)
    er_p_1=deepencoder(n,k,snr1_th)
   # print(er_p_1)

    for i in range(0, num_events):
        er_p = 1- (active_prob * (1 - er_p_1) * ((1 - active_prob) ** (num_nodes - 1)))
        er_indi = int(random.random() > er_p)
        if er_indi == 0:
            departure_timestamps_s[i] = 0
            server_timestamps_1[i] = 0

        else:
            departure_timestamps_s[i] = arrival_timestamps[i] + \
                inter_service_times[i]
            server_timestamps_1[i] = arrival_timestamps[i]
    # print(server_timestamps_1,departure_timestamps_s)
    dep = [x for x in departure_timestamps_s if x != 0]
    sermat = [x for x in server_timestamps_1 if x != 0]
    depcop = dep.copy()
    if server_timestamps_1[-1] == 0:
        if len(depcop) != 0:
            depcop.pop()
            maxt = max(arrival_timestamps[-1], dep[-1])
        else:
            maxt = arrival_timestamps[-1]
        v1 = depcop + [maxt]
    else:
        v1 = dep

    if departure_timestamps_s[0] == 0:
        if len(sermat) != 0:
            t1 = sermat
        else:
            t1 = [0]
    else:
        t1 = [0] + sermat
  # print(sermat, dep)
    system_time = 1 / lambda1  # system time (time which update in the system)
    av_age_simulation, _, _ = average_age_of_information_fn(
        v1, t1, system_time)
    return av_age_simulation

def plot_av_age_simulation(num_nodes, active_prob, n, k, P_range):
    import matplotlib.pyplot as plt
    av_age_simulations_ls = []
    for P in P_range:
        av_age_sim = main(num_nodes, active_prob, n, k, P)
        av_age_simulations_ls.append(av_age_sim)

    plt.plot(P_range, av_age_simulations_ls, marker='o')
    plt.xlabel('Transmission Power')
    plt.ylabel('Average Age of Information')
    plt.title('Average Age of Information vs Transmission Power')
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

num_nodes = 2
active_prob = 0.5
n = 7
k = 4
P_range = [0.01, 0.05, 0.1, 0.5, 1, 5, 10]

plot_av_age_simulation(num_nodes, active_prob, n, k, P_range)

