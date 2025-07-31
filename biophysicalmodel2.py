from neuron import h
import matplotlib.pyplot as plt
import numpy as np
from neuron.units import ms, mV

# Initialize NEURON
h.load_file('stdrun.hoc')
np.random.seed(42)

class SimpleMORSimulation:
    def __init__(self, mor_sensitivity=1.0):
        """
        Fixed MOR simulation with proper network connectivity
        mor_sensitivity: 1.0 = normal, >1.0 = more sensitive, <1.0 = less sensitive
        """
        self.mor_sensitivity = mor_sensitivity
        self.neurons = {}
        self.synapses = {}
        self.stimulators = {}
        self.recordings = {}
        self.netcons = {}
        
    def create_neurons(self):
        """Create the basic circuit: VTA dopamine neuron and GABA interneuron"""
        
        # VTA dopamine neuron (target of disinhibition)
        self.neurons['vta_da'] = h.Section(name='vta_dopamine')
        self.neurons['vta_da'].L = 25
        self.neurons['vta_da'].diam = 25
        self.neurons['vta_da'].Ra = 100
        self.neurons['vta_da'].cm = 1
        self.neurons['vta_da'].insert('hh')
        # Make it more excitable
        self.neurons['vta_da'].gnabar_hh = 0.12
        self.neurons['vta_da'].gkbar_hh = 0.036
        self.neurons['vta_da'].gl_hh = 0.0003
        self.neurons['vta_da'].el_hh = -54.3
        
        # GABA interneuron (where MORs are located)
        self.neurons['gaba'] = h.Section(name='gaba_interneuron')
        self.neurons['gaba'].L = 15
        self.neurons['gaba'].diam = 15
        self.neurons['gaba'].Ra = 100
        self.neurons['gaba'].cm = 1
        self.neurons['gaba'].insert('hh')
        # Make GABA neuron naturally active
        self.neurons['gaba'].gnabar_hh = 0.15
        self.neurons['gaba'].gkbar_hh = 0.03
        self.neurons['gaba'].gl_hh = 0.0005
        self.neurons['gaba'].el_hh = -50
        
        # NAc neuron (receives dopamine)
        self.neurons['nac'] = h.Section(name='nac_neuron')
        self.neurons['nac'].L = 20
        self.neurons['nac'].diam = 20
        self.neurons['nac'].Ra = 100
        self.neurons['nac'].cm = 1
        self.neurons['nac'].insert('hh')
        
    def create_gaba_stimulation(self):
        """Create stimulation to make GABA neuron fire"""
        
        # Excitatory synapse on GABA neuron to make it fire
        gaba_exc_syn = h.ExpSyn(self.neurons['gaba'](0.5))
        gaba_exc_syn.tau = 2
        gaba_exc_syn.e = 0  # Excitatory
        
        # Strong stimulation to make GABA fire
        gaba_stim = h.NetStim()
        gaba_stim.number = 1000  # Many stimuli
        gaba_stim.start = 10
        gaba_stim.interval = 5   # Fast firing
        gaba_stim.noise = 0.3
        
        # Connect stimulation to GABA neuron
        gaba_netcon = h.NetCon(gaba_stim, gaba_exc_syn)
        gaba_netcon.delay = 1
        gaba_netcon.weight[0] = 0.8  # Strong enough to cause firing
        
        self.synapses['stim_to_gaba'] = gaba_exc_syn
        self.stimulators['gaba_stim'] = gaba_stim
        self.netcons['gaba_stim'] = gaba_netcon
        
        return gaba_netcon
    
    def create_gaba_to_da_inhibition(self):
        """Create GABA inhibition of dopamine neuron"""
        
        # GABA synapse on dopamine neuron (inhibitory)
        gaba_syn = h.ExpSyn(self.neurons['vta_da'](0.5))
        gaba_syn.tau = 8
        gaba_syn.e = -80  # Inhibitory
        
        # Connect GABA neuron output to dopamine neuron
        gaba_to_da_netcon = h.NetCon(self.neurons['gaba'](0.5)._ref_v, gaba_syn, 
                                    sec=self.neurons['gaba'])
        gaba_to_da_netcon.threshold = -20  # Spike threshold
        gaba_to_da_netcon.delay = 2
        gaba_to_da_netcon.weight[0] = 3.0  # Strong inhibition
        
        self.synapses['gaba_to_da'] = gaba_syn
        self.netcons['gaba_to_da'] = gaba_to_da_netcon
        
        return gaba_to_da_netcon
    
    def create_mor_mechanism(self, opioid_dose=0.0):
        """
        Create MOR mechanism that inhibits GABA (disinhibition)
        opioid_dose: 0.0 = no opioids, higher values = more opioid effect
        """
        
        if opioid_dose > 0:
            # MOR synapse on GABA neuron (opioids inhibit GABA neurons)
            mor_syn = h.ExpSyn(self.neurons['gaba'](0.5))
            mor_syn.tau = 20  # Slower than regular synapses (opioid effect)
            mor_syn.e = -90   # Very inhibitory
            
            # Opioid stimulation (represents drug binding to MORs)
            mor_stim = h.NetStim()
            mor_stim.number = int(50 * opioid_dose * self.mor_sensitivity)
            mor_stim.start = 200  # Start during "opioid period"
            mor_stim.interval = max(2, 8 / (opioid_dose * self.mor_sensitivity))
            mor_stim.noise = 0.2
            
            # Connect opioid effect to GABA neuron (inhibit GABA)
            mor_netcon = h.NetCon(mor_stim, mor_syn)
            mor_netcon.delay = 1
            mor_netcon.weight[0] = opioid_dose * self.mor_sensitivity * 2.0
            
            self.synapses['mor_to_gaba'] = mor_syn
            self.stimulators['mor'] = mor_stim
            self.netcons['mor_to_gaba'] = mor_netcon
            
            return mor_netcon
        
        return None
    
    def create_dopamine_pathway(self):
        """Create dopamine pathway from VTA to NAc"""
        
        # Dopamine synapse on NAc
        da_syn = h.ExpSyn(self.neurons['nac'](0.5))
        da_syn.tau = 5
        da_syn.e = 0  # Excitatory
        
        # Connect VTA dopamine neuron to NAc
        da_netcon = h.NetCon(self.neurons['vta_da'](0.5)._ref_v, da_syn, 
                            sec=self.neurons['vta_da'])
        da_netcon.threshold = -20
        da_netcon.delay = 2
        da_netcon.weight[0] = 1.5
        
        self.synapses['da_to_nac'] = da_syn
        self.netcons['da_to_nac'] = da_netcon
        
        return da_netcon
    
    def add_vta_stimulation(self):
        """Add some baseline stimulation to VTA to help it fire"""
        
        baseline_syn = h.ExpSyn(self.neurons['vta_da'](0.5))
        baseline_syn.tau = 3
        baseline_syn.e = 0
        
        baseline_stim = h.NetStim()
        baseline_stim.number = 100
        baseline_stim.start = 50
        baseline_stim.interval = 20
        baseline_stim.noise = 0.5
        
        baseline_netcon = h.NetCon(baseline_stim, baseline_syn)
        baseline_netcon.delay = 1
        baseline_netcon.weight[0] = 1.5  # Moderate stimulation
        
        self.stimulators['vta_baseline'] = baseline_stim
        self.synapses['baseline_to_vta'] = baseline_syn
        self.netcons['baseline_to_vta'] = baseline_netcon
    
    def setup_recording(self):
        """Setup recording of neural activity"""
        
        self.recordings['time'] = h.Vector().record(h._ref_t)
        self.recordings['vta_da_v'] = h.Vector().record(self.neurons['vta_da'](0.5)._ref_v)
        self.recordings['gaba_v'] = h.Vector().record(self.neurons['gaba'](0.5)._ref_v)
        self.recordings['nac_v'] = h.Vector().record(self.neurons['nac'](0.5)._ref_v)
        
        # Record synaptic currents
        if 'gaba_to_da' in self.synapses:
            self.recordings['gaba_current'] = h.Vector().record(self.synapses['gaba_to_da']._ref_i)
        
        if 'mor_to_gaba' in self.synapses:
            self.recordings['mor_current'] = h.Vector().record(self.synapses['mor_to_gaba']._ref_i)
        
        if 'da_to_nac' in self.synapses:
            self.recordings['da_current'] = h.Vector().record(self.synapses['da_to_nac']._ref_i)
        
        # Spike detection for all neurons
        self.recordings['vta_spikes'] = h.Vector()
        self.recordings['gaba_spikes'] = h.Vector()
        
        # VTA spike detector
        self.vta_spike_detector = h.NetCon(self.neurons['vta_da'](0.5)._ref_v, None, 
                                          sec=self.neurons['vta_da'])
        self.vta_spike_detector.threshold = -20
        self.vta_spike_detector.record(self.recordings['vta_spikes'])
        
        # GABA spike detector
        self.gaba_spike_detector = h.NetCon(self.neurons['gaba'](0.5)._ref_v, None, 
                                           sec=self.neurons['gaba'])
        self.gaba_spike_detector.threshold = -20
        self.gaba_spike_detector.record(self.recordings['gaba_spikes'])
    
    def run_simulation(self, opioid_dose=0.0, duration=500):
        """
        Run the simulation with specified opioid dose
        
        opioid_dose: 0.0 = baseline, 1.0 = moderate dose, 2.0+ = high dose
        """
        
        print(f"Running simulation with opioid dose: {opioid_dose}")
        print(f"MOR sensitivity: {self.mor_sensitivity}")
        
        # Create network
        self.create_neurons()
        self.create_gaba_stimulation()  # Make GABA fire
        self.create_gaba_to_da_inhibition()  # GABA inhibits DA
        self.create_mor_mechanism(opioid_dose)  # MOR inhibits GABA
        self.create_dopamine_pathway()  # DA to NAc
        self.add_vta_stimulation()  # Help VTA fire
        
        # Setup recording
        self.setup_recording()
        
        # Run simulation
        h.finitialize(-65 * mV)
        h.continuerun(duration * ms)
        
        # Analyze results
        return self.analyze_results(opioid_dose)
    
    def analyze_results(self, opioid_dose):
        """Analyze simulation results"""
        
        # Convert to numpy arrays
        t = np.array(self.recordings['time'])
        vta_v = np.array(self.recordings['vta_da_v'])
        gaba_v = np.array(self.recordings['gaba_v'])
        
        # Get spikes
        vta_spikes = np.array(self.recordings['vta_spikes']) if len(self.recordings['vta_spikes']) > 0 else np.array([])
        gaba_spikes = np.array(self.recordings['gaba_spikes']) if len(self.recordings['gaba_spikes']) > 0 else np.array([])
        
        # Calculate firing rates (split by baseline vs opioid periods)
        baseline_vta = vta_spikes[vta_spikes <= 200] if len(vta_spikes) > 0 else np.array([])
        opioid_vta = vta_spikes[vta_spikes > 200] if len(vta_spikes) > 0 else np.array([])
        
        baseline_gaba = gaba_spikes[gaba_spikes <= 200] if len(gaba_spikes) > 0 else np.array([])
        opioid_gaba = gaba_spikes[gaba_spikes > 200] if len(gaba_spikes) > 0 else np.array([])
        
        # Convert to Hz (spikes per 200ms periods)
        baseline_vta_rate = len(baseline_vta) / 0.2  # Hz
        opioid_vta_rate = len(opioid_vta) / 0.3     # Hz (300ms period)
        baseline_gaba_rate = len(baseline_gaba) / 0.2
        opioid_gaba_rate = len(opioid_gaba) / 0.3
        
        vta_response = opioid_vta_rate - baseline_vta_rate
        gaba_response = opioid_gaba_rate - baseline_gaba_rate
        
        results = {
            'opioid_dose': opioid_dose,
            'mor_sensitivity': self.mor_sensitivity,
            'baseline_vta_rate': baseline_vta_rate,
            'opioid_vta_rate': opioid_vta_rate,
            'baseline_gaba_rate': baseline_gaba_rate,
            'opioid_gaba_rate': opioid_gaba_rate,
            'vta_response': vta_response,
            'gaba_response': gaba_response,
            'time': t,
            'vta_voltage': vta_v,
            'gaba_voltage': gaba_v,
            'vta_spikes': vta_spikes,
            'gaba_spikes': gaba_spikes
        }
        
        print(f"Baseline VTA firing: {baseline_vta_rate:.1f} Hz")
        print(f"Opioid period VTA firing: {opioid_vta_rate:.1f} Hz")
        print(f"VTA response: {vta_response:+.1f} Hz")
        print(f"Baseline GABA firing: {baseline_gaba_rate:.1f} Hz")
        print(f"Opioid period GABA firing: {opioid_gaba_rate:.1f} Hz")
        print(f"GABA response: {gaba_response:+.1f} Hz")
        
        return results
    
    def plot_results(self, results):
        """Plot the simulation results"""
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        
        t = results['time']
        vta_v = results['vta_voltage']
        gaba_v = results['gaba_voltage']
        vta_spikes = results['vta_spikes']
        gaba_spikes = results['gaba_spikes']
        
        # VTA dopamine neuron voltage
        axes[0].plot(t, vta_v, 'r-', linewidth=1.5, label='VTA Dopamine Neuron')
        axes[0].axhline(-20, color='gray', linestyle='--', alpha=0.7, label='Spike Threshold')
        axes[0].axvline(200, color='purple', linestyle=':', linewidth=2, alpha=0.8, label='Opioid Onset')
        
        # Add VTA spikes
        if len(vta_spikes) > 0:
            spike_heights = np.ones_like(vta_spikes) * 30
            axes[0].scatter(vta_spikes, spike_heights, color='red', s=30, alpha=0.8, marker='v')
        
        axes[0].set_ylabel('VTA Voltage (mV)')
        axes[0].set_title(f'MOR Simulation: Dose {results["opioid_dose"]}x, Sensitivity {results["mor_sensitivity"]}x')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(-80, 40)
        
        # GABA neuron voltage
        axes[1].plot(t, gaba_v, 'orange', linewidth=1.5, label='GABA Interneuron')
        axes[1].axhline(-20, color='gray', linestyle='--', alpha=0.7)
        axes[1].axvline(200, color='purple', linestyle=':', linewidth=2, alpha=0.8)
        
        # Add GABA spikes
        if len(gaba_spikes) > 0:
            spike_heights = np.ones_like(gaba_spikes) * 30
            axes[1].scatter(gaba_spikes, spike_heights, color='orange', s=30, alpha=0.8, marker='v')
        
        axes[1].set_ylabel('GABA Voltage (mV)')
        axes[1].set_title('GABA Interneuron (MOR Target)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(-80, 40)
        
        # Synaptic currents
        if 'gaba_current' in self.recordings:
            gaba_i = np.array(self.recordings['gaba_current'])
            axes[2].plot(t, gaba_i, 'blue', linewidth=1.5, label='GABA → VTA Inhibition')
        
        if 'mor_current' in self.recordings:
            mor_i = np.array(self.recordings['mor_current'])
            axes[2].plot(t, mor_i, 'purple', linewidth=1.5, label='MOR → GABA Inhibition')
        
        axes[2].axvline(200, color='purple', linestyle=':', linewidth=2, alpha=0.8)
        axes[2].set_ylabel('Current (nA)')
        axes[2].set_title('Synaptic Currents: MOR Mechanism')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Firing rate summary
        baseline_period = (t >= 50) & (t <= 200)
        opioid_period = (t >= 200) & (t <= 500)
        
        time_bins = np.arange(0, 500, 25)  # 25ms bins
        vta_rate_bins = []
        gaba_rate_bins = []
        
        for i in range(len(time_bins)-1):
            start, end = time_bins[i], time_bins[i+1]
            vta_count = np.sum((vta_spikes >= start) & (vta_spikes < end))
            gaba_count = np.sum((gaba_spikes >= start) & (gaba_spikes < end))
            vta_rate_bins.append(vta_count / 0.025)  # Convert to Hz
            gaba_rate_bins.append(gaba_count / 0.025)
        
        bin_centers = time_bins[:-1] + 12.5
        axes[3].plot(bin_centers, vta_rate_bins, 'r-', linewidth=2, label=f'VTA Dopamine ({results["vta_response"]:+.1f} Hz)')
        axes[3].plot(bin_centers, gaba_rate_bins, 'orange', linewidth=2, label=f'GABA ({results["gaba_response"]:+.1f} Hz)')
        axes[3].axvline(200, color='purple', linestyle=':', linewidth=2, alpha=0.8)
        axes[3].set_xlabel('Time (ms)')
        axes[3].set_ylabel('Firing Rate (Hz)')
        axes[3].set_title('Neural Activity Over Time')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        # Add annotations
        axes[0].text(100, 30, 'BASELINE\n(GABA inhibits VTA)', ha='center', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        axes[0].text(350, 30, 'OPIOID EFFECT\n(MOR inhibits GABA\n→ VTA disinhibited)', ha='center', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
        
        plt.tight_layout()
        plt.show()

def run_dose_response_experiment():
    """Test different opioid doses"""
    print("=== MU-OPIOID RECEPTOR DOSE-RESPONSE EXPERIMENT ===")
    
    doses = [0.0, 0.5, 1.0, 1.5, 2.0]
    results = []
    
    for dose in doses:
        print(f"\n--- Testing Dose: {dose}x ---")
        
        sim = SimpleMORSimulation(mor_sensitivity=1.0)
        result = sim.run_simulation(opioid_dose=dose, duration=500)
        results.append(result)
        
        # Plot key examples
        if dose in [0.0, 2.0]:
            sim.plot_results(result)
    
    # Summary plot
    doses_array = np.array([r['opioid_dose'] for r in results])
    vta_responses = np.array([r['vta_response'] for r in results])
    gaba_responses = np.array([r['gaba_response'] for r in results])
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(doses_array, vta_responses, 'ro-', linewidth=2, markersize=8, label='VTA Dopamine')
    plt.plot(doses_array, gaba_responses, 'o-', color='orange', linewidth=2, markersize=8, label='GABA')
    plt.xlabel('Opioid Dose')
    plt.ylabel('Firing Rate Change (Hz)')
    plt.title('MOR Dose-Response')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    baseline_vta = [r['baseline_vta_rate'] for r in results]
    opioid_vta = [r['opioid_vta_rate'] for r in results]
    
    x = np.arange(len(doses))
    width = 0.35
    
    plt.bar(x - width/2, baseline_vta, width, label='Baseline', alpha=0.7)
    plt.bar(x + width/2, opioid_vta, width, label='With Opioid', alpha=0.7)
    plt.xlabel('Dose Level')
    plt.ylabel('VTA Firing Rate (Hz)')
    plt.title('VTA Dopamine Response')
    plt.xticks(x, [f'{d}x' for d in doses])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    return results

if __name__ == "__main__":
    print("=== FIXED MU-OPIOID RECEPTOR SIMULATION ===")
    print("This simulation shows how MORs mediate opioid effects on dopamine")
    print("Key mechanism: Opioids → MOR activation → GABA inhibition → Dopamine disinhibition")
    
    # Run basic demonstration
    print("\n1. Basic demonstration:")
    
    # No opioids
    print("Testing baseline (no opioids)...")
    sim_baseline = SimpleMORSimulation()
    result_baseline = sim_baseline.run_simulation(opioid_dose=0.0)
    sim_baseline.plot_results(result_baseline)
    
    # High opioid dose
    print("Testing high opioid dose...")
    sim_opioid = SimpleMORSimulation()
    result_opioid = sim_opioid.run_simulation(opioid_dose=2.0)
    sim_opioid.plot_results(result_opioid)
    
    # Run dose-response experiment
    print("\n2. Dose-response experiment:")
    dose_results = run_dose_response_experiment()