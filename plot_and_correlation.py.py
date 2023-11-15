import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# aepe_var = np.load('results/P001/aepe_var.npy')
# aepe_mse = np.load('results/P001/aepe_mse.npy')
# aepe_var = np.array(aepe_var)
# aepe_mse = np.array(aepe_mse)
i = list(range(5001, 120002, 5000))
aepe_mse = np.load('results/eval/5001/aepe_mse.npy')
aepe_mse = np.array(aepe_mse)
aepe_mse = np.mean(aepe_mse, axis=0)
aepe_var_all = []
for j in i:
    aepe_var = np.load('results/eval/' + str(j) + '/aepe_var.npy')
    aepe_var = np.array(aepe_var)
    aepe_var_all.append(aepe_var)
aepe_var_all = np.array(aepe_var_all)
aepe_var = np.mean(aepe_var_all, axis=1)
aepe_mse = np.repeat(aepe_mse[np.newaxis, :], aepe_var.shape[0], axis=0)
cc = []
auc = []
dauc = []
for i in range(aepe_var.shape[0]):
    aepe_mse[i] = (aepe_mse[i] - np.min(aepe_mse[i])) / \
        (np.max(aepe_mse[i]) - np.min(aepe_mse[i]))
    factor = aepe_mse[i][0] / aepe_var[i][0]
    aepe_var[i] = aepe_var[i] * factor
    delta = aepe_var[i] - aepe_mse[i]

    cc.append(spearmanr(aepe_var[i], aepe_mse[i])[0])
    auc.append(np.sum(aepe_var[i]) / aepe_var.shape[1])
    dauc.append(np.sum(aepe_var[i]) / np.sum(aepe_mse[i]))
cc = np.array(cc)
auc = np.array(auc)
dauc = np.array(dauc)
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].plot(cc, label='cc')
axs[0, 0].set_title('CC')
axs[0, 0].legend()
axs[0, 1].plot(auc, label='auc')
axs[0, 1].set_title('AUC')
axs[0, 1].legend()
axs[1, 0].plot(dauc, label='dauc')
axs[1, 0].set_title('DAUC')
axs[1, 0].legend()
for ax in axs:
    for a in ax:
        a.set(xlabel='Epochs', ylabel='Value')

min_cc_index = np.argmin(cc)
max_cc_index = np.argmax(cc)
min_var = aepe_var[min_cc_index]
min_mse = aepe_mse[min_cc_index]
max_var = aepe_var[max_cc_index]
max_mse = aepe_mse[max_cc_index]

mid_var = aepe_var[aepe_var.shape[0] // 2]
mid_mse = aepe_mse[aepe_mse.shape[0] // 2]

indices = np.random.randint(0, aepe_var.shape[0], 5)
x = np.linspace(0, 1, aepe_mse.shape[1])
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'pink']

for index, i in enumerate(indices):

    axs[1, 1].plot(x,
                   aepe_var[i],
                   color=colors[index],
                   alpha=0.8,
                   linewidth=0.7)
    axs[1, 1].plot(x,
                   aepe_mse[i],
                   color=colors[index],
                   alpha=0.8,
                   linewidth=0.7,
                   linestyle='--')
# plt.plot(x, min_var, label='min_var', color='blue', linewidth=2)
# plt.plot(x, max_var, label='max_var', color='red', linewidth=2)
# plt.plot(x,
#          min_mse,
#          label='min_mse',
#          color='blue',
#          linewidth=2,
#          linestyle='--')
# plt.plot(x, max_mse, label='max_mse', color='red', linewidth=2, linestyle='--')
axs[1, 1].set_title('Sparsification Plot')
axs[1, 1].set(xlabel='Pixels removed', ylabel='AEPE')
fig.tight_layout()
plt.savefig('results/eval/common_sparsification_plot.png')
plt.show()
plt.figure()
plt.plot(x, mid_var, label='mid_var', color='red', linewidth=2)
plt.plot(x, mid_mse, label='mid_mse', color='red', linewidth=2, linestyle='--')
plt.savefig('results/eval/common_sparsification_plot.png')
plt.legend(loc='best')
plt.show()
