from neuron import h, gui
import matplotlib.pyplot as plt
import numpy as np
from neuron.units import ms, mV
import os
import random
from scipy.ndimage import gaussian_filter1d


# === Load Mechanisms ===
dll_path = os.path.abspath("./x86_64/.libs/libnrnmech.dylib")
if not os.path.exists(dll_path):
   dll_path = os.path.abspath("./arm64/.libs/libnrnmech.dylib")
h.nrn_load_dll(dll_path)


np.random.seed(42)
random.seed(42)


# === Create Neurons ===
def create_vta_neuron():
   soma = h.Section(name='soma')
   soma.L = soma.diam = 22
   soma.cm = 1.2
   soma.Ra = 150
   soma.insert('pas')
   soma.g_pas = 5e-5
   soma.e_pas = -67 + random.uniform(-1.5, 1.5)
   soma.insert('girk')
   soma.gk_girk_girk = 5e-5
   return soma


def create_nac_neuron():
   soma = h.Section(name='nac')
   soma.L = soma.diam = 20
   soma.insert('hh')
   soma.v = -67 + random.uniform(-1.5, 1.5)
   return soma


# === Synaptic Connection ===
def connect_neurons_with_dopamine(pre, post):
   syn = h.ExpSyn(post(0.5))
   syn.tau = 4
   syn.e = 0
   netcon = h.NetCon(pre(0.5)._ref_v, syn, sec=pre)
   netcon.threshold = -20
   netcon.delay = 1.5 + random.uniform(-0.5, 0.5)
   netcon.weight[0] = max(0.1, np.random.normal(loc=0.8, scale=0.1))
   return syn, netcon


# === Noise Current ===
def add_noise_current(cell, mean=0.002, std=0.001):
   stim = h.IClamp(cell(0.5))
   stim.delay = 0
   stim.dur = 1e9
   stim.amp = max(0, min(0.008, random.gauss(mean, std)))
   return stim


# === Build Population ===
NUM_VTA = 50
NUM_NAC = 50


vta_neurons = []
nac_neurons = []
vta_spikes = []
gaba_netcons = []
baseline_netcons = []
original_gaba_weight = 1.2
original_girk = 5e-5


# === D2 Autoreceptor Parameters ===
d2_feedback_strength = 0.002
syn_da_history = []


for _ in range(NUM_VTA):
   vta = create_vta_neuron()
   add_noise_current(vta)


   baseline_stim = h.NetStim()
   baseline_stim.number = 20
   baseline_stim.start = 180
   baseline_stim.interval = 15
   baseline_stim.noise = 0.5


   exc_syn = h.ExpSyn(vta(0.5))
   exc_syn.tau = 2
   exc_syn.e = 0


   baseline_nc = h.NetCon(baseline_stim, exc_syn)
   baseline_nc.delay = 1
   baseline_nc.weight[0] = 0.6
   baseline_netcons.append(baseline_nc)


   gaba_stim = h.NetStim()
   gaba_stim.number = 40
   gaba_stim.start = 10
   gaba_stim.interval = 8
   gaba_stim.noise = 0.6


   gaba_syn = h.ExpSyn(vta(0.5))
   gaba_syn.tau = 9
   gaba_syn.e = -75


   gaba_nc = h.NetCon(gaba_stim, gaba_syn)
   gaba_nc.delay = 1
   gaba_nc.weight[0] = original_gaba_weight
   gaba_netcons.append(gaba_nc)


   vta_neurons.append(vta)


for _ in range(NUM_NAC):
   nac = create_nac_neuron()
   add_noise_current(nac, mean=0.0015, std=0.0008)
   nac_neurons.append(nac)


# === Recording ===
t = h.Vector().record(h._ref_t)
vta_spikes = []
vta_vm_vectors = []
for vta in vta_neurons:
   spike_vec = h.Vector()
   nc = h.NetCon(vta(0.5)._ref_v, None, sec=vta)
   nc.threshold = -20
   nc.record(spike_vec)
   vta_spikes.append(spike_vec)
   vta_vm_vectors.append(h.Vector().record(vta(0.5)._ref_v))


v_nac = h.Vector().record(nac_neurons[0](0.5)._ref_v)


# === VTA to NAc Synaptic Connections ===
synapses = []
netcons = []
for pre in vta_neurons:
   for post in np.random.choice(nac_neurons, size=2, replace=False):
       syn, nc = connect_neurons_with_dopamine(pre, post)
       synapses.append(syn)
       netcons.append(nc)


# === MOR Modulation ===
def activate_mor():
   print(f">> MOR activated at {h.t}ms - reducing inhibition and GIRK")
   for vta in vta_neurons:
       vta.gk_girk_girk = 1e-5
   for nc in gaba_netcons:
       nc.weight[0] = 0.1  # Stronger inhibition reduction
   for nc in baseline_netcons:
       nc.weight[0] = 2.0  # Stronger excitation


def deactivate_mor():
   print(f">> MOR deactivated at {h.t}ms - restoring inhibition and GIRK")
   for vta in vta_neurons:
       vta.gk_girk_girk = original_girk
   for nc in gaba_netcons:
       nc.weight[0] = original_gaba_weight
   for nc in baseline_netcons:
       nc.weight[0] = 0.6


cvode = h.CVode()
cvode.event(200, activate_mor)
cvode.event(400, deactivate_mor)
cvode.event(600, activate_mor)  # Second opioid administration (delayed to allow recovery)
cvode.event(800, deactivate_mor)  # End second administration (600 + 200 = 800)


# === Run Simulation ===
h.finitialize(-67 * mV)
h.continuerun(1200 * ms)  # Extended to ensure we capture the second opioid period


# === Analyze Spikes ===
t_np = np.array(t)
all_spikes = []
for vec in vta_spikes:
   all_spikes.extend(vec.to_python())
all_spikes = np.array(sorted(all_spikes))


burst_spikes = []
tonic_spikes = []
burst_window_ms = 80
min_burst_spikes = 3


for i in range(len(all_spikes) - min_burst_spikes + 1):
   if all_spikes[i + min_burst_spikes - 1] - all_spikes[i] <= burst_window_ms:
       burst_spikes.extend(all_spikes[i:i + min_burst_spikes])


burst_spikes = np.unique(np.array(burst_spikes))
tonic_spikes = np.setdiff1d(all_spikes, burst_spikes)


# === DA Pools ===
syn_da = np.zeros_like(t_np)
extra_da = np.zeros_like(t_np)


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


for i in range(1, len(t_np)):
   dt = t_np[i] - t_np[i - 1]
   prev_syn = syn_da[i - 1]
   prev_extra = extra_da[i - 1]
   reuptake = Vmax * prev_syn / (Km + prev_syn) * dt
   diffusion = diffusion_rate * prev_syn * dt
   syn_da[i] = max(0, prev_syn + syn_da[i] - reuptake - diffusion)
   extra_da[i] = max(0, prev_extra + extra_da[i] + diffusion - clear_extra_rate * prev_extra * dt)


   # D2 Autoreceptor Feedback
   inhibition = d2_feedback_strength * extra_da[i]
   for vta in vta_neurons:
       vta.g_pas = max(1e-6, 5e-5 * (1 - inhibition))


# === Plot ===
extra_da_smooth = gaussian_filter1d(extra_da, sigma=5)


v_vta = None
for vec in vta_spikes:
   if len(vec) > 0:
       idx = vta_spikes.index(vec)
       v_vta = np.array(vta_vm_vectors[idx])
       break
if v_vta is None:
   v_vta = np.array(vta_vm_vectors[0])


plt.figure(figsize=(14, 8))


plt.subplot(3, 1, 1)
plt.plot(t_np, v_vta, label='VTA Vm (sample)', color='blue')
plt.plot(t_np, v_nac, label='NAc Vm (sample)', color='orange', alpha=0.7)
plt.axvspan(200, 400, color='red', alpha=0.2, label='MOR active (1st)')
plt.axvspan(600, 800, color='red', alpha=0.2, label='MOR active (2nd)')
plt.legend(); plt.grid(True); plt.title("Membrane Potentials (Sample Neurons)")
plt.ylabel("Voltage (mV)")


plt.subplot(3, 1, 2)
plt.plot(t_np, syn_da, label='Synaptic DA', color='purple')
plt.plot(t_np, extra_da_smooth, label='Extrasynaptic DA', color='green')
plt.axvspan(200, 400, color='red', alpha=0.2, label='MOR active (1st)')
plt.axvspan(600, 800, color='red', alpha=0.2, label='MOR active (2nd)')
plt.ylabel("DA Level (a.u.)")
plt.title("Synaptic vs Extrasynaptic Dopamine Pools")
plt.legend(); plt.grid(True)


plt.subplot(3, 1, 3)
if len(burst_spikes) > 0 and len(tonic_spikes) > 0:
   plt.eventplot([burst_spikes, tonic_spikes], colors=['red', 'black'],
                 lineoffsets=[1.2, 0.8], linelengths=[0.3, 0.3])
elif len(burst_spikes) > 0:
   plt.eventplot([burst_spikes], colors=['red'], lineoffsets=[1.0], linelengths=[0.3])
elif len(tonic_spikes) > 0:
   plt.eventplot([tonic_spikes], colors=['black'], lineoffsets=[1.0], linelengths=[0.3])
plt.axvspan(200, 400, color='red', alpha=0.2)
plt.axvspan(600, 800, color='red', alpha=0.2)
plt.ylim(0.5, 1.5)
plt.title("VTA Population Spikes (Burst: red, Tonic: black)")
plt.xlabel("Time (ms)"); plt.grid(True)


plt.tight_layout()
plt.show()


print(f"\nSpike Statistics:")
print(f"Total spikes: {len(all_spikes)}")
print(f"Burst spikes: {len(burst_spikes)}")
print(f"Tonic spikes: {len(tonic_spikes)}")
print(f"Spikes before MOR (0-200ms): {len(all_spikes[all_spikes < 200])}")
print(f"Spikes during 1st MOR (200-400ms): {len(all_spikes[(all_spikes >= 200) & (all_spikes < 400)])}")
print(f"Spikes between MOR periods (400-600ms): {len(all_spikes[(all_spikes >= 400) & (all_spikes < 600)])}")
print(f"Spikes during 2nd MOR (600-800ms): {len(all_spikes[(all_spikes >= 600) & (all_spikes < 800)])}")
print(f"Spikes after 2nd MOR (800-1000ms): {len(all_spikes[all_spikes >= 800])}")
print(f"All spike times: {sorted(all_spikes)}")

