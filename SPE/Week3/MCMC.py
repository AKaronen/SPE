import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.style.use("seaborn")


def dist(sample, sigma=None, step_size=1):
    if sigma is None:
        sigma = np.eye(len(sample)) * 1
    new_sample = sample + (np.random.randn(len(sample)) @ sigma) * step_size
    return new_sample


def MHMC(ssfun, init_state, N, step, burn=0.3):
    burnin = int(burn * N)
    curr_state = init_state
    curr_lh = ssfun(curr_state)
    samples = []
    acc = 0
    for i in range(N):
        proposal_state = dist(curr_state, step_size=step)
        prop_lh = ssfun(proposal_state)
        acc_crit = prop_lh / curr_lh
        acc_threshold = np.random.uniform(0, 1)
        if acc_crit > acc_threshold:
            curr_state = proposal_state
            curr_lh = prop_lh
            if i > burnin:
                acc += 1

        samples.append(curr_state)
    print(f"Accept ratio: {acc/(N-burnin)}")
    return np.array(samples[burnin:]), acc


def adap_MHMC(ssfun, init_state, init_Cov, N, N0, step, burn=0.3):
    burnin = int(burn * N)
    curr_state = init_state
    curr_Cov = init_Cov
    curr_lh = ssfun(curr_state)
    eps = 1e-9
    sd = (2.4**2) / len(init_state)
    samples = []
    acc = 0
    for i in range(N):
        proposal_state = dist(
            curr_state, sigma=np.linalg.cholesky(curr_Cov), step_size=step
        )
        prop_lh = ssfun(proposal_state)
        acc_crit = prop_lh / curr_lh
        acc_threshold = np.random.uniform(0, 1)

        if acc_crit > acc_threshold:
            curr_state = proposal_state
            curr_lh = prop_lh
            if i > burnin:
                acc += 1
        if i >= N0:
            delta = sd * eps * np.identity(len(init_state))
            curr_Cov = sd * np.cov(np.array(samples).T) + delta

        samples.append(curr_state)
    print(f"Accept ratio: {acc/(N-burnin)}")
    return np.array(samples[burnin:]), acc


def DRMC(ssfun, init_state, N, step=1, burn=0.3, m=2):
    burnin = int(0.2 * N)
    curr_state = init_state
    curr_lh = ssfun(curr_state)
    proposal_state = curr_state
    samples = []
    acc = 0
    for i in range(N):
        proposal_state = dist(curr_state, step_size=step)
        prop_lh = ssfun(proposal_state)
        acc_crit = prop_lh / curr_lh
        acc_threshold = np.random.uniform(0, 1)

        if acc_crit > acc_threshold:
            k = 1
            while acc_crit < acc_threshold and k <= m:
                proposal_state = dist(curr_state, step_size=step)
                prop_lh = ssfun(proposal_state)
                acc_crit = prop_lh / curr_lh
                acc_threshold = np.random.uniform(0, 1)
                if acc_crit > acc_threshold:
                    curr_state = proposal_state
                    curr_lh = prop_lh
                    if i > burnin:
                        acc += 1
                k += 1
        else:
            curr_state = proposal_state
            curr_lh = prop_lh
            if i > burnin:
                acc += 1

        samples.append(curr_state)
    print(f"Accept ratio: {acc/(N-burnin)}")
    return np.array(samples[burnin:]), acc


def adap_DRMC(ssfun, init_state, init_Cov, N, N0, step=1, burn=0.3, m=2):
    burnin = int(0.2 * N)
    curr_state = init_state
    curr_Cov = init_Cov
    curr_lh = ssfun(curr_state)
    gamma = 1
    eps = 1e-9
    sd = (2.4**2) / len(init_state)
    samples = []
    acc = 0
    for i in range(N):
        proposal_state = dist(curr_state, curr_Cov, step)
        prop_lh = ssfun(proposal_state)
        acc_crit = prop_lh / curr_lh
        acc_threshold = np.random.uniform(0, 1)
        if (acc_crit <= acc_threshold).all():
            k = 0
            prop_Cov = gamma * curr_Cov

            while (acc_crit <= acc_threshold).all() and k < m:
                proposal_state = dist(curr_state, curr_Cov, step)
                prop_lh = ssfun(proposal_state)
                acc_crit = prop_lh / curr_lh
                acc_threshold = np.random.uniform(0, 1)
                if acc_crit > acc_threshold:
                    curr_state = proposal_state
                    curr_lh = prop_lh
                    if i > burnin:
                        acc += 1
                k += 1
                prop_Cov *= gamma
        else:
            curr_state = proposal_state
            curr_lh = prop_lh
            if i > burnin:
                acc += 1
        if i >= N0:
            delta = sd * eps * np.eye(len(init_state))
            curr_Cov = sd * np.cov(np.array(samples).T) + delta

        samples.append(curr_state)
    print(f"Accept ratio: {acc/(N-burnin)}")
    return np.array(samples[burnin:]), acc


def plot_chains(chains, labels=None):
    nparams = chains.shape[1]
    for i in range(nparams):
        plt.plot(chains[:, i])
        if labels is not None:
            plt.title(labels[i])
        plt.show()


def plot_intervals(lower, upper, estim, obs, model, t, labels=None):
    n_curves = model.shape[1]
    fig = plt.figure(facecolor="w")
    ax = fig.add_subplot(111, facecolor="#dddddd", axisbelow=True)
    ax.plot(t, model, lw=2)
    ax.set_prop_cycle(None)
    ax.plot(t, estim, lw=1, alpha=0.9)
    ax.set_prop_cycle(None)
    ax.plot(t, obs, ls="--")
    if labels is not None:
        ax.legend(labels)
    for i in range(n_curves):
        ax.fill_between(t, lower[:, i], upper[:, i], alpha=0.3)
    ax.grid(which="major", c="w", lw=2, ls="-")
    ax.set_title("Model prediction vs. Ground truth")
    plt.show()
