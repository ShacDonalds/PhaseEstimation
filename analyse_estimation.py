import qsharp
from PhaseEstimation import run
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

phi = 0
nb_run = 20


def estim_phi(n_shots, phi, n_oracle):
    phi_esti = 0
    for _ in range(nb_run):
        result = run.simulate(nShots=n_shots, phi=phi, oraclePower=n_oracle)
        p_esti = result[1] / n_shots
        phi_esti += (2 / n_oracle) * (np.arcsin(np.sqrt(p_esti)) - np.pi / 4)
    return phi_esti / nb_run


nb_range = 200

delta_phi_nshots = []
print("Processing n_shots")
for n in range(1, nb_range):
    delta_phi_nshots.append(abs(phi - estim_phi(n, phi, 10)))

print("Processing n_oracle")
delta_phi_noracles = []
for n in range(10, nb_range):
    delta_phi_noracles.append(abs(phi - estim_phi(10, phi, n)))

plt.plot(range(10, nb_range), delta_phi_nshots, label="nshots")
plt.plot(range(10, nb_range), delta_phi_noracles, label="n_oracles")
plt.title("Comparaison de la precision en fonction de n_shots et n_oracles")
plt.xlabel("n_shots ou n_oracles")
plt.ylabel("precision")
plt.legend()


X, reg_oracles, reg_shots = [], [], []
for i in range(len(delta_phi_noracles)):
    reg_oracles.append(np.log(delta_phi_noracles[i]))
    reg_shots.append(np.log(delta_phi_nshots[i]))
    X.append(np.log(i))

plt.figure()
plt.plot(X, reg_shots, label="nshots")
plt.plot(X, reg_oracles, label="noracles")
plt.title("Regression lineaire pour n_shots et n_oracles")
plt.xlabel("log(n)")
plt.ylabel("-a log(n) + log(b)")
plt.legend()

deg_oracles, _, _, _, _ = stats.linregress(X, reg_oracles)
deg_shots, _, _, _, _ = stats.linregress(X, reg_shots)

print(
    f"Alpha pour la variation d'oracles: {deg_oracles}\nAlpha pour la variation de shots:{deg_shots}"
)
