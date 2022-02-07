import matplotlib.pyplot as plt
import matplotlib, os
import numpy as np

vals = [1, 10, 20, 40, 60, 80, 100, 120, 150, 200, 500, 1000, 5000, 10000, 20000]

labels = list(map(str, vals))

err = []
for el in labels:
    d = np.load('../mdata/ydata-18-01-22-%s.npz'%el)
    x1, y1 = d['y_pred'], d['y_obs']
    er = abs(y1-x1)/y1*100
    err.append(er)
    
labels[-1] = 'All'

merr = list(map(np.mean, err))

ax = plt.gca()

ax.set_xlabel('Number of Cycles Utilised')
ax.set_ylabel('Mean Error Between Observations and Predictions (\%)')

x = np.arange(len(labels))

labels[-1] = 'All'

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation = 90)

ax.bar(x, merr, 0.5, color = 'red', alpha = 0.5)
ax.plot(x, merr, lw = 0.7, color = 'red', marker = '.', markersize = 5)

path = r'D:\WSL\ansto\figs'

plt.savefig(os.path.join(path, 'breaking.pdf'), bbox_inches = 'tight')

plt.show()

#%%

ax = plt.gca()

ax.set_ylabel('Number of Cycles Utilised')
ax.set_xlabel('Error Between Observations and Predictions (\%)')

ax.set_yticklabels(labels)

medianprops = dict(linestyle='-', linewidth=1, color='red')

ax.boxplot(err, vert = False, medianprops = medianprops)

path = r'D:\WSL\ansto\figs'

plt.savefig(os.path.join(path, 'bboxplot.pdf'), bbox_inches = 'tight')

plt.show()