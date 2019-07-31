"""Functions useful for computing the periodicty
"""
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from bpnet.plot.profiles import extract_signal
from bpnet.plot.tracks import plot_tracks
from bpnet.plot.heatmaps import heatmap_contribution_profile, normalize


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def compute_power_spectrum(pattern, task, data):
    seqlets = data.seqlets_per_task[pattern]
    wide_seqlets = [s.resize(data.footprint_width)
                    for s in seqlets
                    if s.center() > data.footprint_width // 2 and
                    s.center() < data.get_seqlen(pattern) - data.footprint_width // 2
                    ]
    p = extract_signal(data.get_region_grad(task, 'profile'), wide_seqlets)

    agg_profile = np.log(np.abs(p).sum(axis=-1).sum(axis=0))

    agg_profile = agg_profile - agg_profile.mean()
    agg_profile = agg_profile / agg_profile.std()

    smooth_part = smooth(agg_profile, 10)
    oscilatory_part = agg_profile - smooth_part

    freq = np.fft.fftfreq(agg_profile[102:].shape[-1])

    freq = freq[:49]
    t0 = 1 / freq
    ps = np.abs(np.fft.fft(oscilatory_part[102:])[:49])**2 + np.abs(np.fft.fft(oscilatory_part[:98])[:49])**2
    return ps, t0, freq
    # plt.savefig('nanog-fft.png', dpi=300)
    # plt.savefig('nanog-fft.pdf')


def periodicity_10bp_frac(pattern, task, data):
    ps, t0, freq = compute_power_spectrum(pattern, task, data)
    assert t0[9] == 10.88888888888889
    return ps[9] / ps.sum()


def plot_power_spectrum(pattern, task, data):
    seqlets = data.seqlets_per_task[pattern]
    wide_seqlets = [s.resize(data.footprint_width)
                    for s in seqlets
                    if s.center() > data.footprint_width // 2 and
                    s.center() < data.get_seqlen(pattern) - data.footprint_width // 2
                    ]
    p = extract_signal(data.get_region_grad(task, 'profile'), wide_seqlets)

    agg_profile = np.log(np.abs(p).sum(axis=-1).sum(axis=0))
    heatmap_contribution_profile(normalize(np.abs(p).sum(axis=-1)[:500], pmin=50, pmax=99), figsize=(10, 20))
    heatmap_fig = plt.gcf()
    # heatmap_contribution_profile(np.abs(p*seq).sum(axis=-1)[:500], figsize=(10, 20))

    agg_profile = agg_profile - agg_profile.mean()
    agg_profile = agg_profile / agg_profile.std()
    freq = np.fft.fftfreq(agg_profile[102:].shape[-1])

    smooth_part = smooth(agg_profile, 10)
    oscilatory_part = agg_profile - smooth_part

    avg_fig, axes = plt.subplots(2, 1, figsize=(11, 4), sharex=True)
    axes[0].plot(agg_profile, label='original')
    axes[0].plot(smooth_part, label="smooth", alpha=0.5)
    axes[0].legend()
    axes[0].set_ylabel("Avg. contribution")
    axes[0].set_title("Average contribution score")
    # axes[0].set_xlabel("Position");
    axes[1].plot(oscilatory_part)
    axes[1].set_xlabel("Position")
    axes[1].set_ylabel("original - smooth")
    avg_fig.subplots_adjust(hspace=0)  # no space between plots
    # plt.savefig('nanog-agg-profile.png', dpi=300)
    # plt.savefig('nanog-agg-profile.pdf')

    fft_fig = plt.figure(figsize=(11, 2))
    plt.plot(1 / freq[:49], np.abs(np.fft.fft(oscilatory_part[102:])[:49])**2 + np.abs(np.fft.fft(oscilatory_part[:98])[:49])**2, "-o")
    plt.xlim([0, 50])
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(25, integer=True))
    plt.grid(alpha=0.3)
    plt.xlabel("1/Frequency [bp]")
    plt.ylabel("Power spectrum")
    plt.title("Power spectrum")
    plt.gcf().subplots_adjust(bottom=0.4)
    return heatmap_fig, avg_fig, fft_fig
