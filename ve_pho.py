import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# nhap duong dan truy suat den thu muc chua cac du lieu cac dot do
path = r'D:\Luan Van Tot Nghiep\Data\Final_Data'
# nhap ten thu muc cua 1 dot do

namefolder = 'Data_ALL'
# nhap cot bat dau ve (cot chua buoc song)
startcol = 10
read_data = pd.read_csv(path + f'\{namefolder}.csv', sep=',')
df = pd.DataFrame(read_data)

# Num1 = df[df['Number of Tangerine'] == 1]
# Num2 = df[df['Number of Tangerine'] == 2]

Position1 = df[df['Point Measurements'] == 1]
Position2 = df[df['Point Measurements'] == 2]
Position3 = df[df['Point Measurements'] == 3]
Position4 = df[df['Point Measurements'] == 4]


def plot(ax, data, start, title):
    y = data.values[:, start - 1:]
    row, _ = y.shape  # so hang cua bo du lieu
    strx = data.columns.values[start - 1:]
    fmx = []
    for x in strx:
        fmx.append(float(x))
    for i in range(0, row):
        ax.plot(fmx, y[i])
    ax.set_xticks(np.arange(int(min(fmx)), int(max(fmx)), 100))
    ax.set_title(title)
    ax.grid()


def plot_mean(data, label, start):
    y = data.values[:, start - 1:]
    ymean = np.mean(y, axis=0)
    strx = data.columns.values[start - 1:]
    fmx = []
    for x in strx:
        fmx.append(float(x))
    plt.plot(fmx, ymean, label=label)
    ax = plt.gca()
    ax.set_xlabel('Wavelegh(nm)', fontsize=15, fontweight='bold')
    ax.set_ylabel('Intensity(a.u)', fontsize=15, fontweight='bold')


# -----------------------------------------------------------------------
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(14, 7))
fig.suptitle('Spectrum ' + namefolder, fontsize=19, fontweight='bold')
plt.subplots_adjust(left=0.076, right=0.96)
plot(ax[0, 0], Position1, start=startcol, title='Point 1')
plot(ax[0, 1], Position2, start=startcol, title='Point 2')
plot(ax[1, 0], Position3, start=startcol, title='Point 3')
plot(ax[1, 1], Position4, start=startcol, title='Point 4')

fig.supxlabel('Wavelegh(nm)', fontsize=15, fontweight='bold')
fig.supylabel('Intensity(a.u)', fontsize=15, fontweight='bold')
plt.savefig(path + '\Spectrum' + '\Spectrum ' + namefolder + '.png')
# -----------------------------------------------------------------------
plt.figure(figsize=(10, 7))
plt.title('Mean Spectrum ' + namefolder, fontsize=19, fontweight='bold')
plot_mean(Position1, start=startcol, label='Point 1')
plot_mean(Position2, start=startcol, label='Point 2')
plot_mean(Position3, start=startcol, label='Point 3')
plot_mean(Position4, start=startcol, label='Point 4')

plt.legend()
plt.grid(1)
plt.savefig(path + '\Spectrum' + '\Mean Spectrum ' + namefolder + '.png')

plt.show()
plt.close('all')
