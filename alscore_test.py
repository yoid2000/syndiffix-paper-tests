import numpy as np
import random
import os
import matplotlib.pyplot as plt
import alscore
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import pprint
pp = pprint.PrettyPrinter(indent=4)

plots_path = os.path.join('als_plots')
if not os.path.exists(plots_path):
    os.makedirs(plots_path)

def plot_abs_weights(out_path):
    als = alscore.ALScore()
    pb_values = np.concatenate([np.arange(0, 0.99, 0.01), np.arange(0.991, 0.999, 0.001)])
    weight_values = [0.5, 1.0, 2.0]
    plt.figure(figsize=((8, 5)))
    for weight in weight_values:
        als.set_param('pcc_abs_weight_strength', weight)
        abs_weight_values = [als._get_pcc_abs_weight(pb) for pb in pb_values]
        plt.plot(pb_values, abs_weight_values, label=f'pcc_abs_weight_strength = {weight}')
    plt.xlim(0.5, 1.0)
    plt.grid(True)
    plt.ylim(0, 1)
    plt.xlabel('PCC', fontsize=12)
    plt.ylabel('abs_weight', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)

def plot_cov_adjust(out_path):
    als = alscore.ALScore()
    ranges = [[0.0001, 0.001], [0.001, 0.01], [0.01, 0.1], [0.1, 1]]
    arrays = [np.linspace(start, end, 1000) for start, end in ranges]
    cov_values = np.concatenate(arrays)
    strength_vals = [1.5, 2.0, 3.0]

    fig, ax1 = plt.subplots(figsize=((8, 5)))

    for n in strength_vals:
        als.set_param('cov_adjust_strength', n)
        adj_values = [als._cov_adjust(cov) for cov in cov_values]
        ax1.scatter(cov_values, adj_values, label=f'cov_adjust_strength = {n}', s=5)

    ax1.set_xscale('log')  # Set the scale of the second x-axis to logarithmic
    ax1.set_xlabel('COV (Log Scale)', fontsize=12)
    ax1.set_ylabel('Adjustment', fontsize=12)

    ax2 = ax1.twiny()  # Create a second x-axis
    #ax2.set_xscale('log')  # Set the scale of the second x-axis to logarithmic

    for n in strength_vals:
        als.set_param('cov_adjust_strength', n)
        adj_values = [als._cov_adjust(cov) for cov in cov_values]
        ax2.scatter(cov_values, adj_values, label=f'cov_adjust_strength = {n}', s=5)

    ax2.set_xlabel('COV (Linear Scale)', fontsize=12)
    ax2.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(out_path)

def plot_base_adjusted_pcc(out_path):
    als = alscore.ALScore()
    increase_values = [0.2, 0.5, 0.8, 0.98]
    pcc_base_values = np.linspace(0, 0.999, 1000)
    fig, ax = plt.subplots(figsize=((8, 5)))
    # For each increase value, calculate pcc_attack and pcc_adj for each pcc_base and plot the results
    for increase in increase_values:
        pcc_attack_values = pcc_base_values + increase * (1.0 - pcc_base_values)
        pcc_adj_values = [als._pcc_improve(pcc_base, pcc_attack) for pcc_base, pcc_attack in zip(pcc_base_values, pcc_attack_values)]
        ax.plot(pcc_base_values, pcc_adj_values, label=f'Improvement = {increase}')

    # Add labels and a legend
    ax.set_ylim(0, 1)
    ax.set_xlabel('Base PCC', fontsize=12)
    ax.set_ylabel('Attack PCC', fontsize=12)
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 0.25))

    # Create an inset axes in the upper left corner of the current axes
    ax_inset = inset_axes(ax, width="30%", height="30%", loc=2, borderpad=4)

    # Plot the same data on the inset axes with the specified x-axis range
    for increase in increase_values:
        pcc_attack_values = pcc_base_values + increase * (1.0 - pcc_base_values)
        pcc_adj_values = [als._pcc_improve(pcc_base, pcc_attack) for pcc_base, pcc_attack in zip(pcc_base_values, pcc_attack_values)]
        ax_inset.plot(pcc_base_values, pcc_adj_values)

    # Set the x-axis range of the inset axes
    ax_inset.set_xlim(0.94, 1.0)
    plt.tight_layout()
    plt.savefig(out_path)

def plot_identical_cov(out_path, limit=1.0):
    ''' In this plot, we hold the precision improvement of the attack over the base constant, and given both attack and base identical coverage. We find that the
    constant precision improvement puts an upper bound on the ALS. We also find that
    the coverage also places an upper bound.
    '''
    als = alscore.ALScore()
    cov_values = np.logspace(np.log10(0.0001), np.log10(1), 5000)
    p_base_values = np.random.uniform(0, limit, len(cov_values))

    # Run several different relative improvements between base and attack
    increase_values = [0.2, 0.5, 0.8, 0.98]
    plt.figure(figsize=((8, 5)))
    for increase in increase_values:
        p_attack_values = p_base_values + (increase * (1.0 - p_base_values))
        scores = [als.alscore(p_base_value, cov_value, p_attack_value, cov_value) for p_base_value, cov_value, p_attack_value, cov_value in zip(p_base_values, cov_values, p_attack_values, cov_values)]
        plt.scatter(cov_values, scores, label=f'precision increase = {increase}', s=2)
    plt.xscale('log')
    plt.grid(True)
    plt.ylim(0, 1)
    plt.xlabel(f'Coverage (base precision limit = {limit})', fontsize=12)
    plt.ylabel('Anonymity Loss Score', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)

def plot_varying_base_coverage(out_path):
    ''' The purpose of this plot is to see the effect of having a different
        base coverage than attack coverage. We vary the base coverage from 1/10K to 1 while keeping all other parameters constant. What this shows is that the ALS varies substantially when the coverage values are not similar.
    '''
    als = alscore.ALScore()
    cov_values = np.logspace(np.log10(0.0001), np.log10(1), 5000)
    p_base = 0.5
    c_attack = 0.01

    # Run several different relative improvements between base and attack
    increase_values = [0.2, 0.5, 0.8, 0.98]
    plt.figure(figsize=((8, 5)))
    for increase in increase_values:
        p_attack = p_base + (increase * (1.0 - p_base))
        scores = [als.alscore(p_base, cov_value, p_attack, c_attack) for cov_value in cov_values]
        plt.scatter(cov_values, scores, label=f'precision increase = {increase}', s=2)
    plt.xscale('log')
    plt.axvline(x=0.01, color='black', linestyle='dashed')
    plt.grid(True)
    plt.ylim(0, 1)
    plt.xlabel(f'Base Coverage (Attack Coverage = {c_attack})', fontsize=12)
    plt.ylabel('Anonymity Loss Score', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)

def do_als_test(als, p_base, c_base, increase, c_attack):
    print('------------------------------------')
    p_attack = p_base + increase * (1.0 - p_base)
    print(f'Base precision: {p_base}, base coverage: {c_base}\nattack precision: {p_attack}, attack coverage: {c_attack}')
    print(f'increase: {increase}')
    print(f'ALS: {round(als.alscore(p_base, c_base, p_attack, c_attack),3)}')

als = alscore.ALScore()
do_als_test(als, p_base=0.5, c_base=1.0, increase=0.2, c_attack=1.0)
do_als_test(als, p_base=0.2, c_base=1.0, increase=0.8, c_attack=1.0)
do_als_test(als, p_base=0.999, c_base=1.0, increase=0.9, c_attack=1.0)
do_als_test(als, p_base=0.5, c_base=0.1, increase=0.2, c_attack=0.1)
do_als_test(als, p_base=0.2, c_base=0.1, increase=0.8, c_attack=0.1)
do_als_test(als, p_base=0.5, c_base=0.01, increase=0.2, c_attack=0.01)
do_als_test(als, p_base=0.2, c_base=0.01, increase=0.8, c_attack=0.01)
do_als_test(als, p_base=0.5, c_base=0.001, increase=0.2, c_attack=0.001)
do_als_test(als, p_base=0.2, c_base=0.001, increase=0.8, c_attack=0.001)
plot_varying_base_coverage(os.path.join(plots_path, 'varying_base_coverage.png'))
plot_identical_cov(os.path.join(plots_path, 'identical_cov.png'))
plot_identical_cov(os.path.join(plots_path, 'identical_cov_limit.png'), limit=0.5)
plot_abs_weights(os.path.join(plots_path, 'abs_weights.png'))
plot_cov_adjust(os.path.join(plots_path, 'cov_adjust.png'))
plot_base_adjusted_pcc(os.path.join(plots_path, 'base_adjusted_pcc.png'))
pass