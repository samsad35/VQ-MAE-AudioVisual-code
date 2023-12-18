import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set_style("darkgrid")
plt.style.use('fivethirtyeight')


fontsize = 28
fontsize2 = 14
linewidth = 2

ratio = [10, 20, 30, 40, 50, 60, 70, 80, 90]

vqmae_av_mean = np.array([30.21294, 29.529734, 28.74743, 28.260925, 26.995974, 26.706423, 25.441648, 23.809538, 21.927012])
vqmae_av_std = np.array([2.427744, 2.4832828, 2.6784754, 2.6299884, 3.09169, 2.5710812, 3.0608864, 2.6789346, 2.8985746])

vqmae_v_mean = np.array([29.71091, 29.321732, 28.669614, 27.830513, 27.010952, 25.715357, 23.8874, 21.362907, 18.246043])
vqmae_v_std = np.array([2.830214, 2.8141646, 2.5991762, 2.3457022, 2.5483143, 3.0230725, 2.8093104, 3.1135645, 2.7637374])

fig, (ax1) = plt.subplots(1, 1, figsize=(10, 5))
# plt.suptitle("Mouth perturbation", fontsize=16)

ax1.plot(ratio, vqmae_av_mean, color='red', label="VQ-MAE-AV-12", linewidth=linewidth)
ax1.fill_between(ratio, vqmae_av_mean - vqmae_av_std, vqmae_av_mean + vqmae_av_std,
                 alpha=0.1, facecolor='red', edgecolor='red')

ax1.plot(ratio, vqmae_v_mean, color='green', label="VQ-MAE-V-12", linewidth=linewidth)
ax1.fill_between(ratio, vqmae_v_mean - vqmae_v_std, vqmae_v_mean + vqmae_v_std,
                 alpha=0.1, facecolor='green', edgecolor='green')

ax1.set_xlabel('Masking ratio (%)', fontsize=fontsize2)
ax1.set_ylabel('Peak Signal-to-Noise Ratio (PSNR in dB)', fontsize=fontsize2)

ax1.legend()
fig.tight_layout()

plt.show()



"""----------------------------------------------------------------------------------------------------------------- """



vqmae_av_mean = np.array([18.850471, 13.880963, 11.523433, 9.647458, 8.672624, 7.1100445, 5.466481, 3.2298086, 0.9390036])
vqmae_av_std = np.array([3.2055564, 2.7486465, 2.213903, 2.1857283, 1.6460369, 1.8682919, 1.669611, 2.0458565, 1.7962968])

vqmae_v_mean = np.array([18.59742, 13.772661, 11.5980015, 9.20134, 7.233931, 4.674451, 2.3218465, 0.355773, -1.933146])
vqmae_v_std = np.array([3.1972938, 2.6045187, 2.4659185, 1.7890028, 1.7974635, 1.6628752, 1.7063226, 1.4913888, 1.7663459])

fig, (ax1) = plt.subplots(1, 1, figsize=(10, 5))
# plt.suptitle("Mouth perturbation", fontsize=16)

ax1.plot(ratio, vqmae_av_mean, color='red', label="VQ-MAE-AV-12", linewidth=linewidth)
ax1.fill_between(ratio, vqmae_av_mean - vqmae_av_std, vqmae_av_mean + vqmae_av_std,
                 alpha=0.1, facecolor='red', edgecolor='red')

ax1.plot(ratio, vqmae_v_mean, color='green', label="VQ-MAE-A-12", linewidth=linewidth)
ax1.fill_between(ratio, vqmae_v_mean - vqmae_v_std, vqmae_v_mean + vqmae_v_std,
                 alpha=0.1, facecolor='green', edgecolor='green')

ax1.set_xlabel('Masking ratio (%)', fontsize=fontsize2)
ax1.set_ylabel('Signal Distortion Ratio (SDR in dB)', fontsize=fontsize2)

ax1.legend()
fig.tight_layout()

plt.show()



fontsize = 28
fontsize2 = 14
linewidth = 1.5

# Create some mock data
performance_h = [80.8, 81.5, 79.5, 77.7]
performance_d = [78.1, 81.5, 81.3, 75.7]
h = [3, 4, 6, 8]
d = [2, 4, 8, 16]

fig, ax1 = plt.subplots(figsize=(4, 10))

color = 'tab:red'
ax1.set_xlabel('Accuracy (%)', fontsize=fontsize2)
ax1.set_ylabel('Size of ($w, h$)', color=color, fontsize=fontsize2)
ax1.plot(performance_h, h, color=color, linewidth=linewidth, marker='X', linestyle='dashed', label="Discrete visual index tokens")
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Size of ($d$)', color=color,  fontsize=fontsize2)  # we already handled the x-label with ax1
ax2.plot(performance_d, d, color=color, linewidth=linewidth, marker='X', linestyle='dashed', label="Discrete audio index tokens")
ax2.tick_params(axis='y', labelcolor=color)

fig.legend(loc="upper right")

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.legend()
plt.show()
