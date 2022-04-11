import matplotlib.pyplot as plt
import matplotlib, os
import numpy as np

vals = [1, 10, 20, 40, 60, 80, 100, 120, 150, 200, 500, 1000, 5000, 10000, 20000]
vals = [1, 5, 10, 60, 120, 500, 1000, 5000, 10834]

vals = [1, 10, 60, 120, 500, 1000, 5000]

labels = list(map(str, vals))

err0 = []; err1= []
for el in labels:
    d = np.load('../mdata/ydata-13-02-22-D-%s.npz'%el)
    x0, y0 = d['y_pred_train'], d['y_obs_train']
    x1, y1 = d['y_pred_test'], d['y_obs_test']
    er0 = abs(y0-x0)/y0*100
    er1 = abs(y1-x1)/y1*100
    err0.append(er0)
    err1.append(er1)
    
labels[-1] = 'All'

merr0 = list(map(np.mean, err0))
merr1 = list(map(np.mean, err1))

ax = plt.gca()

ax.set_xlabel('Number of Cycles Utilised')
ax.set_ylabel('Mean Error Between Observations and Predictions (\%)')

x = np.arange(len(labels))

labels[-1] = 'All'

ax.set_xticks(x)
ax.set_xticklabels(labels)

w = 0.3

ax.bar(x-w/2, merr0, w, color = 'red', alpha = 0.5, label = 'Training Data')
ax.plot(x-w/2, merr0, lw = 0.7, color = 'red', marker = '.', markersize = 5)

ax.bar(x+w/2, merr1, w, color = 'blue', alpha = 0.5, label = 'Test Data')
ax.plot(x+w/2, merr1, lw = 0.7, color = 'blue', marker = '.', markersize = 5)

path = r'D:\WSL\ansto\figs'

ax.legend(framealpha = 1, edgecolor = 'None', loc = 0)

# plt.savefig(os.path.join(path, 'breaking3.pdf'), bbox_inches = 'tight')

plt.show()

#%%

ax = plt.gca()

ax.set_ylabel('Number of Cycles Utilised')
ax.set_xlabel('Error Between Observations and Predictions (\%)')

ax.set_yticklabels(labels)

medianprops = dict(linestyle='-', linewidth=1, color='red')

ax.boxplot(err1, vert = False, medianprops = medianprops)

path = r'D:\WSL\ansto\figs'

# plt.savefig(os.path.join(path, 'bboxplot.pdf'), bbox_inches = 'tight')

plt.show()

print(np.mean(merr0), np.mean(merr1))