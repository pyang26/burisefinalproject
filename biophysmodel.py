from neuron import h, gui
import matplotlib.pyplot as plt
import numpy as np
from neuron.units import ms, mV
import os
import random

# === Load Mechanisms ===
dll_path = os.path.abspath("./x86_64/.libs/libnrnmech.dylib")
if not os.path.exists(dll_path):
   dll_path = os.path.abspath("./arm64/.libs/libnrnmech.dylib")
h.nrn_load_dll(dll_path)

np.random.seed(42)
random.seed(42)

# === Create Biologically Grounded VTA Neuron ===
def create_vta_neuron():
   soma = h.Section(name='soma')
   soma.L = soma.diam = 22
   soma.cm = 1.2  # slightly higher capacitance for mammalian neurons
   soma.Ra = 150
   soma.insert('pas')
   soma.g_pas = 5e-5  # physiological input resistance
   soma.e_pas = -67 + random.uniform(-1.5, 1.5)
   soma.insert('girk')
   soma.gk_girk_girk = 5e-5  # basal GIRK conductance
   return soma

# === Create NAc Neuron ===
def create_nac_neuron():
   soma = h.Section(name='nac')
   soma.L = soma.diam = 20
   soma.insert('hh')
   soma.v = -67 + random.uniform(-1.5, 1.5)
   return soma

# === Dopaminergic Synaptic Connection ===
def connect_neurons_with_dopamine(pre, post):
   syn = h.ExpSyn(post(0.5))
   syn.tau = 4
   syn.e = 0
   netcon = h.NetCon(pre(0.5)._ref_v, syn, sec=pre)
   netcon.threshold = -20
   netcon.delay = 1.5 + random.uniform(-0.5, 0.5)
   netcon.weight[0] = max(0.1, np.random.normal(loc=0.8, scale=0.1))
   return syn, netcon

vta = create_vta_neuron()
nac = create_nac_neuron()

# === Noise Current ===
def add_noise_current(cell, mean=0.004, std=0.0015):
   stim = h.IClamp(cell(0.5))
   stim.delay = 0
   stim.dur = 1e9
   stim.amp = max(0, min(0.012, random.gauss(mean, std)))
   return stim

add_noise_current(vta)
add_noise_current(nac, mean=0.0025)

# === Baseline and MOR Stimulation Adjustments ===
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

# === GABA Inhibition ===
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

# === MOR Mechanism ===
original_gaba_weight = gaba_netcon.weight[0]
original_girk = vta.gk_girk_girk

def simulate_mor():
   gaba_netcon.weight[0] = 0.1
   vta.gk_girk_girk = 2e-5
   opioid_netcon.weight[0] *= 1.6
   print(">> MOR activated")

def end_mor():
   gaba_netcon.weight[0] = original_gaba_weight
   vta.gk_girk_girk = original_girk
   opioid_netcon.weight[0] /= 1.6
   print(">> MOR deactivated")

cvode = h.CVode()
cvode.event(200, simulate_mor)
cvode.event(400, end_mor)

# === Dopamine Synapse ===
da_syn, da_netcon = connect_neurons_with_dopamine(vta, nac)

# === Recording ===
t = h.Vector().record(h._ref_t)
v_vta = h.Vector().record(vta(0.5)._ref_v)
v_nac = h.Vector().record(nac(0.5)._ref_v)
i_da = h.Vector().record(da_syn._ref_i)
i_gaba = h.Vector().record(gaba_syn._ref_i)

spikes = h.Vector()
spike_detector = h.NetCon(vta(0.5)._ref_v, None, sec=vta)
spike_detector.threshold = -20
spike_detector.record(spikes)

    # === Run Simulation (First pass, no feedback) ===
h.finitialize(-65 * mV)
h.continuerun(1000 * ms)

# This is the updated simulation integrating:
# - Burst vs Tonic DA release
# - D2 autoreceptor feedback (on g_pas & GIRK)
# - Synaptic vs Extrasynaptic DA pools
# - Michaelis-Menten DAT reuptake kinetics
# This patch replaces your dopamine dynamics and feedback section.

# === Analyze Spikes ===
t_np = np.array(t)
spikes_np = np.array(spikes)

# Burst vs Tonic Classification
burst_spikes = []
tonic_spikes = []
burst_window_ms = 80
min_burst_spikes = 3

for i in range(len(spikes_np) - min_burst_spikes + 1):
   if spikes_np[i + min_burst_spikes - 1] - spikes_np[i] <= burst_window_ms:
      burst_spikes.extend(spikes_np[i:i + min_burst_spikes])

burst_spikes = np.unique(np.array(burst_spikes))
tonic_spikes = np.setdiff1d(spikes_np, burst_spikes)

# === Dual DA Pools ===
syn_da = np.zeros_like(t_np)
extra_da = np.zeros_like(t_np)

# Parameters
release_tonic = 1.0
release_burst = 3.0

Vmax = 0.5
Km = 5.0
diffusion_rate = 0.01
clear_extra_rate = 0.002

burst_idx = np.searchsorted(t_np, burst_spikes)
tonic_idx = np.searchsorted(t_np, tonic_spikes)

for idx in burst_idx:
   if idx < len(syn_da):
      syn_da[idx] += release_burst
for idx in tonic_idx:
   if idx < len(syn_da):
      syn_da[idx] += release_tonic

# Dynamics Loop
for i in range(1, len(t_np)):
   dt = t_np[i] - t_np[i - 1]
   prev_syn = syn_da[i - 1]
   prev_extra = extra_da[i - 1]

   reuptake = Vmax * prev_syn / (Km + prev_syn) * dt
   diffusion = diffusion_rate * prev_syn * dt

   syn_da[i] = max(0, prev_syn + syn_da[i] - reuptake - diffusion)
   extra_da[i] = max(0, prev_extra + extra_da[i] + diffusion - clear_extra_rate * prev_extra * dt)

# === D2 Feedback ===
gpas_base = 5e-5
girk_base = 5e-5
gpas_mod = []
girk_mod = []

for da_val in extra_da:
   if da_val > 3.0:
      inhib = min(1.0, (da_val - 3.0) / 5.0)
      gpas_mod.append(gpas_base * (1 - 0.5 * inhib))
      girk_mod.append(girk_base * (1 + 2 * inhib))
   else:
      gpas_mod.append(gpas_base)
      girk_mod.append(girk_base)

# === Apply Feedback ===
gpas_vec = h.Vector(gpas_mod)
girk_vec = h.Vector(girk_mod)
gpas_vec.play(vta(0.5)._ref_g_pas, 1)
girk_vec.play(vta(0.5)._ref_gk_girk_girk, 1)

# === Replace DA Plot ===
from scipy.ndimage import gaussian_filter1d
extra_da_smooth = gaussian_filter1d(extra_da, sigma=5)

#plot 1
plt.figure(figsize=(14, 10))
plt.subplot(4, 1, 1)
plt.plot(t_np, v_vta, label='VTA Vm', color='blue')
plt.plot(t_np, v_nac, label='NAc Vm', color='orange', alpha=0.7)
plt.axvspan(200, 400, color='red', alpha=0.2)
plt.legend(); plt.grid(True); plt.title("Membrane Potentials")

#plot 2
# plt.subplot(4, 1, 2)
# plt.eventplot([spikes_np], colors=['black'], lineoffsets=[1], linelengths=[0.8])
# plt.axvspan(200, 400, color='red', alpha=0.2)
# plt.title("Spike Raster (VTA)"); plt.ylim(0.5, 1.5); plt.grid(True)

#plot 3
plt.subplot(4, 1, 3)
plt.plot(t_np, syn_da, label='Synaptic DA', color='purple')
plt.plot(t_np, extra_da_smooth, label='Extrasynaptic DA', color='green')
plt.axvspan(200, 400, color='red', alpha=0.2)
plt.ylabel("DA Level (a.u.)")
plt.title("Synaptic vs Extrasynaptic Dopamine Pools")
plt.legend(); plt.grid(True)

#plot 4
# plt.subplot(4, 1, 4)
# plt.plot(t_np, gpas_mod, label='g_pas (D2)', color='black')
# plt.plot(t_np, girk_mod, label='g_GIRK (D2)', color='teal')
# plt.xlabel("Time (ms)"); plt.ylabel("Conductance (S/cm^2)")
# plt.title("D2 Autoreceptor Feedback")
# plt.legend(); plt.grid(True)

plt.tight_layout()
plt.show()

# Function to get dopamine data for integration
def get_dopamine_data():
    """Return the dopamine data from the biophysical simulation for integration."""
    # This function should be called after the simulation has run
    return {
        'time': t_np,
        'synaptic_da': syn_da,
        'extrasynaptic_da': extra_da,
        'total_da': syn_da + extra_da,
        'opioid_active': [1.0 if 200 <= t <= 400 else 0.0 for t in t_np]
    }