import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.stats import spearmanr


def sparsification_plot(var, mse):
    '''
    step:
    1:sort each pixel's variance from large to small and get the index
    2:remove 10 pixels each time and calculate the remaining pixels' mean mse
    3:plot the mean mse with the percentage of removed pixels
    '''
    var = var.flatten()
    mse = mse.flatten()
    sorted_indices = np.argsort(var)[::-1]
    sorted_indices_mse = np.argsort(mse)[::-1]
    mean_mses = []
    mean_mses_mse = []
    for i in range(0, var.shape[0], var.shape[0] // 1000):
        remain_indices = sorted_indices[i:]
        remain_indices_mse = sorted_indices_mse[i:]
        if remain_indices.shape[0] == 0:
            mean_mses.append(0)
            mean_mses_mse.append(0)
            break
        mean_mses.append(np.mean(mse[remain_indices]))
        mean_mses_mse.append(np.mean(mse[remain_indices_mse]))

    return mean_mses, mean_mses_mse


if __name__ == '__main__':
    i = list(range(5001, 120002, 5000))
    cov_path = []
    for j in i:
        cov_path.append('results/eval/' + str(j) + '/cov/file/')
    mse_path = 'results/eval/5001/mse/file/'
    mselist = sorted(glob.glob(mse_path + '*.npy'))
    mses = []
    for mse in mselist:
        mses.append(np.load(mse))
    mses = np.array(mses)
    for covdir in cov_path:
        covlist = sorted(glob.glob(covdir + '*.npy'))
        covs = []
        for cov in covlist:
            covs.append(np.load(cov))
        covs = np.array(covs)
        aepe_cov = []
        aepe_mse = []
        cc = []
        for i in range(len(covlist)):
            print('processing {}th image'.format(i))
            mean_mses, mean_mses_mse = sparsification_plot(covs[i], mses[i])
            aepe_cov.append(mean_mses)
            aepe_mse.append(mean_mses_mse)
            var = covs[i].flatten()
            mse = mses[i].flatten()
            cc.append(spearmanr(var, mse)[0])
        aepe_cov = np.array(aepe_cov)
        aepe_mse = np.array(aepe_mse)
        cc = np.array(cc)

        plt.plot(cc, label='cc')
        plt.plot([np.mean(cc)] * len(cc), label='mean')
        plt.legend()
        plt.title('Spearman’s rank correlation of variance and mse')
        plt.savefig('results/eval/' + covdir.split('/')[2] + '/cov&mse_cc.png')
        plt.close()
        aepe_cov = np.array(aepe_cov)
        aepe_mse = np.array(aepe_mse)
        #将所有nan替换为0
        aepe_cov = np.nan_to_num(aepe_cov)
        aepe_mse = np.nan_to_num(aepe_mse)
        np.save('results/eval/' + covdir.split('/')[2] + '/aepe_var.npy',
                aepe_cov)
        np.save('results/eval/' + covdir.split('/')[2] + '/aepe_mse.npy',
                aepe_mse)
