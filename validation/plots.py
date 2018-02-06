#!/usr/bin/env python

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


SIZES = [32, 64, 100, 128, 200, 256, 300]
SIZES_DISP = [32, 128, 200, 256, 300]
FILENAME_TEMPLATE = "lid-driven-cavity_{n}x{n}.json"

DATASET_TR1920X_VS_GTX760 = [
    ("AMD TR-1920X (12-Kern CPU)", "./data/TR1920X"),
    ("NVIDIA GTX-760", "./data/GTX760"),
]
DATASET_TR1920X_VS_GTX760_RELBASE = "AMD TR-1920X (12-Kern CPU)"


def load_json(path):
    with open(path) as f:
        return json.load(f)

def get_results_avg(sizes, directory, key):
    data = []
    for n in sizes:
        filename = directory + '/' + FILENAME_TEMPLATE.format(n=n)
        entry = load_json(filename)[key]
        data.append(entry["duration"] / entry["executions"])

    return [x / 1000.0 for x in data]

def calc_speedup(base, other):
    return np.divide(base, other)


def plot_speedup(datasets, relbase):
    results_tts = {k : get_results_avg(SIZES, v, "tts::full") for k, v in datasets}
    results_solver = {k : get_results_avg(SIZES, v, "solver::iteration::full") for k, v in datasets}

    base_tts = results_tts[relbase]
    other_tts = [(k, v) for k, v in results_tts.items() if k != relbase]
    speedup_tts = [(k, calc_speedup(base_tts, o)) for k, o in other_tts]

    base_solver = results_solver[relbase]
    other_solver = [(k, v) for k, v in results_solver.items() if k != relbase]
    speedup_solver = [(k, calc_speedup(base_solver, o)) for k, o in other_solver]

    fig, ax = plt.subplots()

    fig.suptitle("Performance-Measurements: Lid-Driven Cavity Flow")
    ax.set_title("Speed-up vs. {}".format(relbase))
    ax.set_xlabel("Size ($N \\times N$)")
    ax.set_ylabel("Speed-Up")

    xs = [x**2 for x in SIZES]

    for k, ys in speedup_tts:
        plt.plot(xs, ys, label="All")

    for k, ys in speedup_solver:
        plt.plot(xs, ys, label="Solver (avg.)")

    ax.set_xticks([x**2 for x in SIZES_DISP])
    ax.set_xticklabels(["${}^2$".format(x) for x in SIZES_DISP])
    ax.legend(loc='best')
    fig.savefig("SpeedUp-GTX760-TR1920X.pdf")


if __name__ == '__main__':
    sns.set()
    plot_speedup(DATASET_TR1920X_VS_GTX760, DATASET_TR1920X_VS_GTX760_RELBASE)
    plt.show()
