from neuron import h, gui
import matplotlib.pyplot as plt
import numpy as np
from neuron.units import ms, mV
import os
import random
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ==================== NEURON SIMULATION AND DATA COLLECTION ====================

def run_simulation_and_get_data():
    """
    Runs the NEURON simulation of the VTA-NAc circuit and collects all relevant
    data for later plotting and analysis.
    """
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)

    # === Try to Load Custom Mechanisms ===
    def load_mechanisms():
        dll_paths = [
            "./x86_64/.libs/libnrnmech.dylib",
            "./arm64/.libs/libnrnmech.dylib",
            "./x86_64/.libs/libnrnmech.so",
            "./libnrnmech.dll"
        ]
        
        for dll_path in dll_paths:
            if os.path.exists(dll_path):
                try:
                    h.nrn_load_dll(dll_path)
                    print(f"Loaded mechanisms from: {dll_path}")
                    return True
                except:
                    continue
        print("Warning: Custom mechanisms not loaded. Using built-in mechanisms.")
        return False

    has_custom_mechanisms = load_mechanisms()

    girk_conductances = {}

    def create_vta_neuron():
        soma = h.Section(name='soma')
        soma.L = soma.diam = 22
        soma.cm = 1.2
        soma.Ra = 150
        soma.insert('pas')
        soma.g_pas = 5e-5
        soma.e_pas = -67 + random.uniform(-1.5, 1.5)
        
        if has_custom_mechanisms:
            try:
                soma.insert('girk')
                soma.gk_girk_girk = 5e-5
            except:
                girk_conductances[soma] = 5e-5
        else:
            girk_conductances[soma] = 5e-5
        return soma

    def create_nac_neuron():
        soma = h.Section(name='nac')
        soma.L = soma.diam = 20
        soma.insert('hh')
        soma.v = -67 + random.uniform(-1.5, 1.5)
        return soma

    def connect_neurons(pre, post):
        syn = h.ExpSyn(post(0.5))
        syn.tau = 4
        syn.e = 0
        netcon = h.NetCon(pre(0.5)._ref_v, syn, sec=pre)
        netcon.threshold = -20
        netcon.delay = 1.5 + random.uniform(-0.5, 0.5)
        netcon.weight[0] = max(0.1, np.random.normal(0.8, 0.1))
        return syn, netcon

    def add_noise(cell, mean=0.004, std=0.0015):
        stim = h.IClamp(cell(0.5))
        stim.delay = 0
        stim.dur = 1e9
        stim.amp = max(0, min(0.012, random.gauss(mean, std)))
        return stim

    vta = create_vta_neuron()
    nac = create_nac_neuron()
    add_noise(vta)
    add_noise(nac, mean=0.0025)

    baseline_stim = h.NetStim()
    baseline_stim.number = 30
    baseline_stim.start = 60
    baseline_stim.interval = 20
    baseline_stim.noise = 0.7
    baseline_syn = h.ExpSyn(vta(0.5))
    baseline_syn.tau = 2
    baseline_syn.e = 0
    baseline_netcon = h.NetCon(baseline_stim, baseline_syn)
    baseline_netcon.delay = 1
    baseline_netcon.weight[0] = 0.7

    opioid_stim = h.NetStim()
    opioid_stim.number = 60
    opioid_stim.start = 200
    opioid_stim.interval = 4
    opioid_stim.noise = 0.85
    opioid_syn = h.ExpSyn(vta(0.5))
    opioid_syn.tau = 2
    opioid_syn.e = 0
    opioid_netcon = h.NetCon(opioid_stim, opioid_syn)
    opioid_netcon.delay = 1
    opioid_netcon.weight[0] = 2.7

    gaba_stim = h.NetStim()
    gaba_stim.number = 30
    gaba_stim.start = 10
    gaba_stim.interval = 9
    gaba_stim.noise = 0.6
    gaba_syn = h.ExpSyn(vta(0.5))
    gaba_syn.tau = 9
    gaba_syn.e = -75
    gaba_netcon = h.NetCon(gaba_stim, gaba_syn)
    gaba_netcon.delay = 1
    gaba_netcon.weight[0] = 2.2

    original_gaba_weight = gaba_netcon.weight[0]
    has_girk_attr = hasattr(vta, 'gk_girk_girk')

    def simulate_mor():
        gaba_netcon.weight[0] = 0.1
        if has_custom_mechanisms:
            try:
                vta.gk_girk_girk = 2e-5
            except:
                pass
        opioid_netcon.weight[0] *= 1.6
        print(">> MOR activated")

    def end_mor():
        gaba_netcon.weight[0] = original_gaba_weight
        if has_custom_mechanisms:
            try:
                vta.gk_girk_girk = 5e-5
            except:
                pass
        opioid_netcon.weight[0] /= 1.6
        print(">> MOR deactivated")

    cvode = h.CVode()
    cvode.event(200, simulate_mor)
    cvode.event(400, end_mor)

    da_syn, da_netcon = connect_neurons(vta, nac)

    t = h.Vector().record(h._ref_t)
    v_vta = h.Vector().record(vta(0.5)._ref_v)
    v_nac = h.Vector().record(nac(0.5)._ref_v)
    i_gaba = h.Vector().record(gaba_syn._ref_i)
    i_da = h.Vector().record(da_syn._ref_i)
    spikes = h.Vector()
    spike_detector = h.NetCon(vta(0.5)._ref_v, None, sec=vta)
    spike_detector.threshold = -20
    spike_detector.record(spikes)

    print("Starting simulation...")
    h.finitialize(-65 * mV)
    h.continuerun(1000 * ms)
    print("Simulation completed")

    t_np = np.array(t)
    spikes_np = np.array(spikes)
    i_gaba_np = np.array(i_gaba)

    def classify_spikes(spike_times, burst_window=80, min_spikes=3):
        if len(spike_times) < min_spikes:
            return np.array([]), spike_times
        burst_spikes = []
        for i in range(len(spike_times) - min_spikes + 1):
            if spike_times[i + min_spikes - 1] - spike_times[i] <= burst_window:
                burst_spikes.extend(spike_times[i:i + min_spikes])
        burst_spikes = np.unique(burst_spikes)
        tonic_spikes = np.setdiff1d(spike_times, burst_spikes)
        return burst_spikes, tonic_spikes

    burst_spikes, tonic_spikes = classify_spikes(spikes_np)
    print(f"Total: {len(spikes_np)}, Burst: {len(burst_spikes)}, Tonic: {len(tonic_spikes)}")

    def simulate_dopamine_dynamics():
        syn_da = np.zeros_like(t_np)
        extra_da = np.zeros_like(t_np)
        release_tonic, release_burst = 1.0, 3.0
        Vmax, Km = 0.5, 5.0
        diffusion_rate, clearance_rate = 0.01, 0.002
        burst_idx = np.searchsorted(t_np, burst_spikes)
        tonic_idx = np.searchsorted(t_np, tonic_spikes)
        for idx in burst_idx:
            if idx < len(syn_da):
                syn_da[idx] += release_burst
        for idx in tonic_idx:
            if idx < len(syn_da):
                syn_da[idx] += release_tonic
        for i in range(1, len(t_np)):
            dt = t_np[i] - t_np[i-1]
            reuptake = Vmax * syn_da[i-1] / (Km + syn_da[i-1]) * dt
            diffusion = diffusion_rate * syn_da[i-1] * dt
            syn_da[i] = max(0, syn_da[i-1] + syn_da[i] - reuptake - diffusion)
            extra_da[i] = max(0, extra_da[i-1] + extra_da[i] + diffusion -
                             clearance_rate * extra_da[i-1] * dt)
        return syn_da, extra_da

    syn_da, extra_da = simulate_dopamine_dynamics()
    extra_da_smooth = gaussian_filter1d(extra_da, sigma=5)

    return {
        't': t_np,
        'v_vta': np.array(v_vta),
        'v_nac': np.array(v_nac),
        'spikes': spikes_np,
        'syn_da': syn_da,
        'extra_da_smooth': extra_da_smooth,
        'i_gaba': i_gaba_np
    }

# ==================== COMBINED VISUALIZATION FUNCTION ====================

def create_combined_visualization(data):
    """
    Creates a single figure with a color-coded heatmap table of neural activity.
    """
    t_np = data['t']
    spikes_np = data['spikes']
    syn_da = data['syn_da']
    extra_da_smooth = data['extra_da_smooth']
    i_gaba_np = data['i_gaba']

    # Define time periods for each state
    time_periods = {
        'Initial State\n(0-150ms)': (0, 150),
        'Opioid Onset\n(150-250ms)': (150, 250),
        'Peak Effect\n(250-350ms)': (250, 350),
        'Withdrawal Phase\n(350-500ms)': (350, 500),
        'Post-Opioid\n(500-1000ms)': (500, 1000)
    }

    # Initialize data storage for the table
    table_data = {
        'Overall Dopamine Output (a.u.)': [],
        'VTA Firing Rate (Hz)': [],
        'GABA Inhibition (pA)': []
    }

    # Calculate average values for each state
    for period in time_periods.values():
        mask = (t_np >= period[0]) & (t_np < period[1])
        if np.any(mask):
            firing_rate = len(spikes_np[(spikes_np >= period[0]) & (spikes_np < period[1])]) * (1000 / (period[1] - period[0]))
            table_data['VTA Firing Rate (Hz)'].append(firing_rate)
            
            gaba_avg = np.mean(np.abs(i_gaba_np[mask]))
            table_data['GABA Inhibition (pA)'].append(gaba_avg)
            
            da_avg = np.mean(syn_da[mask] + extra_da_smooth[mask])
            table_data['Overall Dopamine Output (a.u.)'].append(da_avg)
        else:
            table_data['VTA Firing Rate (Hz)'].append(0)
            table_data['GABA Inhibition (pA)'].append(0)
            table_data['Overall Dopamine Output (a.u.)'].append(0)
            
    # Normalize each metric row independently from 0 to 1
    normalized_matrix = np.zeros((len(table_data), len(time_periods)))
    row_keys = list(table_data.keys())
    for i, metric in enumerate(row_keys):
        row_values = np.array(table_data[metric])
        min_val = np.min(row_values)
        max_val = np.max(row_values)
        
        if (max_val - min_val) > 0:
            normalized_matrix[i, :] = (row_values - min_val) / (max_val - min_val)
        else:
            normalized_matrix[i, :] = 0

    # Invert the GABA Inhibition row: high GABA inhibition = low neural activity (low color value)
    gaba_row_index = row_keys.index('GABA Inhibition (pA)')
    normalized_matrix[gaba_row_index, :] = 1 - normalized_matrix[gaba_row_index, :]

    # Set up the plot for the heatmap table
    fig, ax = plt.subplots(figsize=(12, 5)) 

    # --- Heatmap Table ---
    cmap = plt.cm.viridis
    im = ax.imshow(normalized_matrix, aspect='auto', cmap=cmap, interpolation='nearest') 

    ax.set_xticks(np.arange(len(time_periods)))
    ax.set_yticks(np.arange(len(row_keys)))
    ax.set_xticklabels(list(time_periods.keys()), fontsize=10)
    ax.set_yticklabels(row_keys, fontsize=10)
    ax.tick_params(top=False, bottom=False, left=False, right=False)
    ax.set_title('Normalized Neural Activity', fontsize=16, pad=20)

    # Add lighter gray lines between cells for better definition
    for x in np.arange(0.5, len(time_periods) - 0.5, 1):
        ax.axvline(x, color='gray', lw=0.5)
    for y in np.arange(0.5, len(row_keys) - 0.5, 1):
        ax.axhline(y, color='gray', lw=0.5)

    # Create and position a horizontal color bar
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', shrink=0.7, pad=0.1)
    cbar.set_label('Normalized Activity (0 = Minimal, 1 = Enhanced)')

    plt.tight_layout()
    plt.show()

# ==================== MAIN EXECUTION BLOCK ====================

if __name__ == '__main__':
    # Run the simulation and get the data
    simulation_data = run_simulation_and_get_data()
    # Call the function to create the color-coded table
    create_combined_visualization(simulation_data)