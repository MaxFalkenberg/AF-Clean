import cPickle
import seaborn as sns
import os

filename = raw_input('Robust data to plot: ')
with open(os.path.join('Data', '%s.p' % filename)) as f:
    sorted_mean_importance, sorted_std_importance, feature_names_sorted = cPickle.load(f)

fig = sns.plt.figure()
sns.set(style="white")
sns.barplot(sorted_mean_importance, feature_names_sorted, xerr=sorted_std_importance, color='r',
            error_kw={'ecolor': 'k'})
sns.plt.title("Feature Importance for %s" % filename, fontsize=16)
sns.plt.xlabel("Mean Feature Importance (Gini)", fontsize=14)
sns.despine(left=True, offset=10)
sns.plt.xlim(left=0)
sns.plt.show()
sns.plt.close()
