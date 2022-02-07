import matplotlib.pyplot as plt
import matplotlib, os
import numpy as np
import pandas as pd

vals = [1, 10, 20, 40, 60, 80, 100, 120, 150, 200, 500, 1000, 5000, 10000, 20000]
vals = [1, 5, 10, 60, 120, 500, 1000, 5000, 10834]

vals = [1, 2, 3, 4, 5] + list(range(10, 4695, 5))

labels = list(map(str, vals))

err0 = []; err1= []
for el in labels:
    d = np.load('../mdata/ydata-03-02-22-%s.npz'%el)
    x0, y0 = d['y_pred_train'], d['y_obs_train']
    x1, y1 = d['y_pred_test'], d['y_obs_test']
    er0 = abs(y0-x0)/y0*100
    er1 = abs(y1-x1)/y1*100
    err0.append(er0)
    err1.append(er1)
    
labels[-1] = 'All'

merr0 = np.array(list(map(np.mean, err0))).reshape(-1, 1)
merr1 = np.array(list(map(np.mean, err1))).reshape(-1, 1)

err = pd.DataFrame(np.concatenate((merr0, merr1), axis = 1), columns = ['merr0', 'merr1'])

avg_err = err.rolling(70, center=True, min_periods=1).mean()

ax = plt.gca()

ax.set_xlabel('Number of Cycles Utilised')
ax.set_ylabel('MAPE (\%)')

ax.set_ylim(10, 40)

ax.plot(vals, merr0, lw = 0.7, color = 'red', alpha = 0.2)
ax.plot(vals, merr1, lw = 0.7, color = 'blue', alpha = 0.2)

ax.plot(vals, avg_err.merr0, lw = 1.5, color = 'red', label = 'Training Data')
ax.plot(vals, avg_err.merr1, lw = 1.5, color = 'blue', label = 'Test Data')

path = r'D:\WSL\ansto\figs'

ax.legend(framealpha = 1, edgecolor = 'None', loc = 0)

# plt.savefig(os.path.join(path, 'breaking3.pdf'), bbox_inches = 'tight')

plt.show()

