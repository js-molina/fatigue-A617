import matplotlib.pyplot as plt
import matplotlib, os
import numpy as np
import pandas as pd

vals = [1, 10, 20, 40, 60, 80, 100, 120, 150, 200, 500, 1000, 5000, 10000, 20000]
vals = [1, 5, 10, 60, 120, 500, 1000, 5000, 10834]

# vals = [1, 2, 3, 4, 5] + list(range(10, 10835, 5))

vals = [5, 10, 50] + list(range(100, 1300, 100))

# vals = [5, 10, 50, 100] + list(range(500, 10900, 500))

# vals = [1, 10, 50, 100, 500] + list(range(1000, 12000, 1000))

labels = list(map(str, vals))

err0 = []; err1= []; err2 = []
for el in labels:
    # d = np.load('../mdata/ydata-18-04-22-D-%s.npz'%el)
    d = np.load('../mdata/ydata-19-06-22-D-1-%s.npz'%el)
    # d = np.load('../mdata/ydata-16-06-22-R-2-%s.npz'%el)
    # d = np.load('../mdata/ydata-11-05-22-D-%s.npz'%el)
    # x0, y0 = d['y_pred_train'][:22], d['y_pred_train'][22:]
    x0, y0 = d['y_pred_train'], d['y_obs_train']
    x1, y1 = d['y_pred_dev'], d['y_obs_dev']
    x2, y2 = d['y_pred_test'], d['y_obs_test']
    er0 = abs(y0-x0)/y0*100
    er1 = abs(y1-x1)/y1*100
    er2 = abs(y2-x2)/y2*100
    err0.append(er0)
    err1.append(er1)
    err2.append(er2)
    
labels[-1] = 'All'

merr0 = np.array(list(map(np.mean, err0))).reshape(-1, 1)
merr1 = np.array(list(map(np.mean, err1))).reshape(-1, 1)
merr2 = np.array(list(map(np.mean, err2))).reshape(-1, 1)

err = pd.DataFrame(np.concatenate((merr0, merr1, merr2), axis = 1), columns = ['merr0', 'merr1', 'merr2'])

avg_err = err.rolling(10, center=True, min_periods=1).mean()

fig, ax = plt.subplots(1, 1, figsize=(4,4))

ax.set_xlabel('Number of Cycles Utilised')
ax.set_ylabel('MAPE (\%)')

ax.set_ylim(0, 40)
ax.set_xlim(0, 10834)

ax.plot(vals, merr0, lw = 0.7, color = 'blue', alpha = 0.5)
ax.plot(vals, merr1, lw = 0.7, color = 'xkcd:green', alpha = 0.5)
ax.plot(vals, merr2, lw = 0.7, color = 'xkcd:red', alpha = 0.5)

ax.plot(vals, avg_err.merr0, lw = 1.5, color = 'blue', label = 'Training Data')
ax.plot(vals, avg_err.merr1, lw = 1.5, color = 'xkcd:green', label = 'Development Data')
ax.plot(vals, avg_err.merr2, lw = 1.5, color = 'red', label = 'Test Data')

path = r'D:\INDEX\TextBooks\Thesis\Engineering\Manuscript\Figures'

ax.legend(framealpha = 1, edgecolor = 'None', loc = 0)

ax.grid(dashes = (1, 5), color = 'gray', lw = 0.7)

# plt.savefig(os.path.join(path, 'breaking_dev_d.pdf'), bbox_inches = 'tight')

plt.show()

