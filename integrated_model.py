#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Import from existing models
from model import DopamineCenteredProcessor, DopamineInput, DopamineScenario
# Import only what we need from model.py

class IntegratedOpioidRewardModel:
    def __init__(self):
        self.biophys_processor = None
        self.reward_processor = DopamineCenteredProcessor(state_dim=8)
        
        # Integrated data storage
        self.opioid_timeline = []
        self.dopamine_timeline = []
        self.rpe_timeline = []
        self.time_points = []
        
        # Opioid simulation parameters
        self.opioid_dose = 1.0  # Arbitrary units
        self.opioid_start_time = 200  # ms
        self.opioid_peak_time = 350   # ms
        self.opioid_end_time = 500   # ms
        
    def run_biophysical_simulation(self):
        """Run the biophysical model to get dopamine output."""
        print("Running biophysical simulation...")
        
        # Import the biophysical model's dopamine calculation directly
        # This simulates the exact same calculation as biophysmodel.py
        time_points = np.linspace(0, 1000, 1000)  # 1000ms simulation
        syn_da = np.zeros_like(time_points)
        extra_da = np.zeros_like(time_points)
        
        # Parameters matching biophysical model exactly
        release_tonic = 1.0
        release_burst = 3.0
        Vmax = 0.3  # Slower reuptake for gradual clearance
        Km = 5.0
        diffusion_rate = 0.005  # Slower diffusion for gradual clearance
        clear_extra_rate = 0.001  # Slower clearance for gradual decrease
        
        # Simulate spikes (like biophysical model)
        # Generate deterministic spikes with opioid modulation
        spikes = []
        opioid_active = []
        
        # Set random seed for reproducible results
        np.random.seed(42)
        
        # Initialize cumulative opioid levels
        cumulative_opioid = 0.0
        opioid_clearance_rate = 0.05  # Rate at which opioids are cleared from system (faster clearance)
        
        for i, t in enumerate(time_points):
            # Calculate rate of opioid input (current implementation)
            opioid_input_rate = 0.0
            opioid_multiplier = 1.0
            
            # Define multiple administration times (in ms)
            admin_times = [200, 550]  # Multiple doses, more spaced out
            admin_duration = 125  # Each dose lasts 150ms
            
            for admin_time in admin_times:
                if admin_time <= t < admin_time + admin_duration:
                    # Rate of opioid input during administration
                    opioid_input_rate = self.opioid_dose  # Rate of 1.0 during administration
                    opioid_multiplier = 1.0 + 3.0  # Constant strong effect
                    break  # Only one administration active at a time
            
            # Integrate rate of change to get total opioid levels
            # Add input rate and subtract clearance rate
            cumulative_opioid += opioid_input_rate - (cumulative_opioid * opioid_clearance_rate)
            cumulative_opioid = max(0.0, cumulative_opioid)  # Can't go below zero
            
            opioid_active.append(cumulative_opioid)
            
            # Continuous dopamine release during opioid administration
            # Dopamine builds up gradually during opioid presence
            if cumulative_opioid > 0:
                # Dopamine release proportional to cumulative opioid level
                # More opioids in system = more dopamine release
                release_amount = 0.001 * opioid_multiplier * cumulative_opioid  # Proportional to opioid level
                
                # Add continuous release to synaptic dopamine
                syn_da[i] += release_amount
        
        # Convert spikes to arrays (like biophysical model)
        spikes_np = np.array(spikes)
        
        # Burst vs Tonic Classification (like biophysical model)
        burst_spikes = []
        tonic_spikes = []
        burst_window_ms = 80
        min_burst_spikes = 3
        
        for i in range(len(spikes_np) - min_burst_spikes + 1):
            if spikes_np[i + min_burst_spikes - 1] - spikes_np[i] <= burst_window_ms:
                burst_spikes.extend(spikes_np[i:i + min_burst_spikes])
        
        burst_spikes = np.unique(np.array(burst_spikes))
        tonic_spikes = np.setdiff1d(spikes_np, burst_spikes)
        
        # Continuous dopamine release is already handled in the loop above
        # No additional spike-based release needed
        
        # Dynamics Loop (exactly like biophysical model)
        for i in range(1, len(time_points)):
            dt = time_points[i] - time_points[i - 1]
            prev_syn = syn_da[i - 1]
            prev_extra = extra_da[i - 1]
            
            reuptake = Vmax * prev_syn / (Km + prev_syn) * dt
            diffusion = diffusion_rate * prev_syn * dt
            
            syn_da[i] = max(0, prev_syn + syn_da[i] - reuptake - diffusion)
            extra_da[i] = max(0, prev_extra + extra_da[i] + diffusion - clear_extra_rate * prev_extra * dt)
        
        # Total dopamine = synaptic + extrasynaptic (exactly like biophysical model)
        total_da = syn_da + extra_da
        dopamine_levels = total_da.tolist()
        return time_points, dopamine_levels, opioid_active
    
    def run_reward_processing(self, time_points, dopamine_levels, opioid_active):
        print("Running reward processing simulation...")
        
        # Set random seed for reproducible reinforcement learning
        np.random.seed(42)
        
        reward_results = []
        
        for i, (t, da_level, opioid_level) in enumerate(zip(time_points, dopamine_levels, opioid_active)):
            # Normalize dopamine to match RL model's expected range (0.1-0.9)
            # The RL model expects dopamine levels similar to its generate_dopamine_scenario method
            # Scale biophysical dopamine to match RL model's dopamine range
            normalized_da = np.clip(0.1 + (da_level / 2.0) * 0.8, 0.1, 0.9)  # Map to RL model range
            
            # Create dopamine input for reward model
            dopamine_input = DopamineInput(
                level=normalized_da,
                scenario=DopamineScenario.NORMAL_DOPAMINE,
                timestamp=i
            )
            # Process reward step using the reinforcement learning model
            # This uses TD learning, dopamine modulation, and state value updates
            # Set random seed for deterministic state transitions
            np.random.seed(42 + i)  # Different seed for each step but deterministic
            
            # Create opioid-based actual reward
            # Higher opioid levels = higher actual reward
            base_reward = 0.3  # Base reward without opioids
            # Scale opioid level to reasonable range (0-1) for reward calculation
            scaled_opioid = np.clip(opioid_level / 20.0, 0.0, 1.0)  # Scale down high opioid levels
            opioid_reward_boost = scaled_opioid * 0.7  # Opioids increase actual reward
            actual_reward = base_reward + opioid_reward_boost
            
            # Create custom reward data for the RL model
            from model import RewardData
            reward_data = RewardData(
                actual_reward=actual_reward,
                expected_reward=0.0,  # Will be calculated by RL model
                timestamp=i
            )
            
            # Process with custom reward data
            # We need to bypass the RL model's reward generation and use our opioid-based reward
            result = self._process_with_custom_reward(dopamine_input, reward_data)
            reward_results.append(result)
            
            # Store integrated data
            self.opioid_timeline.append(opioid_level)
            self.dopamine_timeline.append(da_level)
            self.rpe_timeline.append(result['rpe'])
            self.time_points.append(t)
            
            #debugging
            if i % 200 == 0:
                print(f"Step {i}: DA={normalized_da:.3f}, Opioid={opioid_level:.3f}, Actual={result['actual_reward']:.3f}, Expected={result['expected_reward']:.3f}, RPE={result['rpe']:.3f}")
        return reward_results
    
    def _process_with_custom_reward(self, dopamine_input, reward_data):
        # Generate expected reward from dopamine input
        expected_reward_data = self.reward_processor.generate_rewards_from_dopamine(dopamine_input, dopamine_input.timestamp, self.reward_processor.rpe_history)
        # Update next state value (simulate environment transition)
        self.reward_processor.next_state_value = self.reward_processor.current_state_value + np.random.normal(0, 0.1)
        # Compute temporal difference RPE with dopamine modulation
        td_rpe = reward_data.actual_reward + self.reward_processor.gamma * self.reward_processor.next_state_value - self.reward_processor.current_state_value
        dopamine_modulation = 1.0 + dopamine_input.level * 0.5
        modulated_td_rpe = td_rpe * dopamine_modulation
        
        # Update state value using temporal difference learning
        self.reward_processor.current_state_value = self.reward_processor.current_state_value + self.reward_processor.eta * modulated_td_rpe
        
        # Use temporal difference RPE as the main RPE
        rpe = modulated_td_rpe
        expected_reward = expected_reward_data.expected_reward
        
        # Record history
        self.reward_processor.actual_rewards.append(reward_data.actual_reward)
        self.reward_processor.expected_rewards.append(expected_reward)
        self.reward_processor.rpe_history.append(rpe)
        self.reward_processor.dopamine_history.append(dopamine_input.level)
        self.reward_processor.state_values.append(self.reward_processor.current_state_value)
        
        return {
            'actual_reward': reward_data.actual_reward,
            'expected_reward': expected_reward,
            'rpe': rpe,
            'dopamine_level': dopamine_input.level
        }
    
    def run_integrated_simulation(self):
        print("=== INTEGRATED OPIOID-DOPAMINE-REWARD SIMULATION ===")
        
        # Step 1: Run biophysical simulation
        time_points, dopamine_levels, opioid_active = self.run_biophysical_simulation()
        
        # Step 2: Run reward processing with dopamine output
        reward_results = self.run_reward_processing(time_points, dopamine_levels, opioid_active)
        
        # Step 3: Analyze and visualize results
        self.analyze_integrated_results(time_points, dopamine_levels, opioid_active, reward_results)
        
        return time_points, dopamine_levels, opioid_active, reward_results
    
    def analyze_integrated_results(self, time_points, dopamine_levels, opioid_active, reward_results):
        # Extract RPE data
        rpe_values = [result['rpe'] for result in reward_results]
        expected_rewards = [result['expected_reward'] for result in reward_results]
        actual_rewards = [result['actual_reward'] for result in reward_results]
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Opioid administration timeline (total levels))
        ax1 = plt.subplot(4, 2, 1)
        ax1.plot(time_points, opioid_active, 'red', linewidth=3)
        ax1.fill_between(time_points, opioid_active, alpha=0.3, color='red')
        ax1.set_title('Total Opioid Levels', fontweight='bold')
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Total Opioid Level (a.u.)')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 50.0) 
        
        # 2. Dopamine response to opioids
        ax2 = plt.subplot(4, 2, 2)
        ax2.plot(time_points, dopamine_levels, 'purple', linewidth=3)
        ax2.axvspan(self.opioid_start_time, self.opioid_end_time, color='red', alpha=0.2)
        ax2.set_title('Dopamine Response to Opioids', fontweight='bold')
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Dopamine Level (a.u.)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Reward processing components
        ax3 = plt.subplot(4, 2, 3)
        ax3.plot(time_points, actual_rewards, 'g-', linewidth=2, label='Actual Reward')
        ax3.plot(time_points, expected_rewards, 'b--', linewidth=2, label='Expected Reward')
        ax3.axvspan(self.opioid_start_time, self.opioid_end_time, color='red', alpha=0.2)
        ax3.set_title('Reinforcement Learning: TD Reward Processing', fontweight='bold')
        ax3.set_xlabel('Time (ms)')
        ax3.set_ylabel('Reward Value (a.u.)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. RPE over time
        ax4 = plt.subplot(4, 2, 4)
        ax4.plot(time_points, rpe_values, 'r-', linewidth=2)
        ax4.axvspan(self.opioid_start_time, self.opioid_end_time, color='red', alpha=0.2)
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax4.set_title('TD Learning: Reward Prediction Error', fontweight='bold')
        ax4.set_xlabel('Time (ms)')
        ax4.set_ylabel('RPE (a.u.)')
        ax4.grid(True, alpha=0.3)
        
        # 5. Opioid → Dopamine correlation
        ax5 = plt.subplot(4, 2, 5)
        ax5.scatter(opioid_active, dopamine_levels, alpha=0.6, s=20, color='purple')
        ax5.set_title('Opioid → Dopamine Correlation', fontweight='bold')
        ax5.set_xlabel('Opioid Level')
        ax5.set_ylabel('Dopamine Level')
        ax5.grid(True, alpha=0.3)
        
        # 6. Dopamine → RPE correlation
        ax6 = plt.subplot(4, 2, 6)
        ax6.scatter(dopamine_levels, rpe_values, alpha=0.6, s=20, color='orange')
        ax6.set_title('Dopamine → RPE Correlation', fontweight='bold')
        ax6.set_xlabel('Dopamine Level')
        ax6.set_ylabel('RPE')
        ax6.grid(True, alpha=0.3)
        
        # 7. Complete pathway: Opioid → RPE
        ax7 = plt.subplot(4, 2, 7)
        ax7.scatter(opioid_active, rpe_values, alpha=0.6, s=20, color='red')
        ax7.set_title('Opioid → RPE Correlation', fontweight='bold')
        ax7.set_xlabel('Opioid Level')
        ax7.set_ylabel('RPE')
        ax7.grid(True, alpha=0.3)
        
        # 8. Temporal alignment of all components
        ax8 = plt.subplot(4, 2, 8)
        ax_twin = ax8.twinx()
        
        # Plot opioid on primary axis
        line1 = ax8.plot(time_points, opioid_active, 'red', linewidth=3, label='Opioid')
        ax8.set_xlabel('Time (ms)')
        ax8.set_ylabel('Opioid Level', color='red')
        ax8.tick_params(axis='y', labelcolor='red')
        ax8.grid(True, alpha=0.3)
        
        # Plot RPE on secondary axis
        line2 = ax_twin.plot(time_points, rpe_values, 'blue', linewidth=2, label='RPE')
        ax_twin.set_ylabel('RPE', color='blue')
        ax_twin.tick_params(axis='y', labelcolor='blue')
        
        ax8.set_title('Opioid → RPE Temporal Alignment', fontweight='bold')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax8.legend(lines, labels, loc='upper right')
        
        plt.suptitle('Integrated Opioid-Dopamine-Reward Processing Pathway', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Print analysis
        self._print_integrated_analysis(time_points, dopamine_levels, opioid_active, rpe_values)
    
    def _print_integrated_analysis(self, time_points, dopamine_levels, opioid_active, rpe_values):
        opioid_da_corr = np.corrcoef(opioid_active, dopamine_levels)[0, 1]
        da_rpe_corr = np.corrcoef(dopamine_levels, rpe_values)[0, 1]
        opioid_rpe_corr = np.corrcoef(opioid_active, rpe_values)[0, 1]
        
        print(f"\n1. Pathway Correlations:")
        print(f"   Opioid → Dopamine: {opioid_da_corr:.3f}")
        print(f"   Dopamine → RPE: {da_rpe_corr:.3f}")
        print(f"   Opioid → RPE: {opioid_rpe_corr:.3f}")
        
        # Analyze opioid effect periods
        opioid_period = (time_points >= self.opioid_start_time) & (time_points <= self.opioid_end_time)
        baseline_period = (time_points < self.opioid_start_time) | (time_points > self.opioid_end_time)
        
        if np.any(opioid_period) and np.any(baseline_period):
            baseline_da = np.mean(np.array(dopamine_levels)[baseline_period])
            opioid_da = np.mean(np.array(dopamine_levels)[opioid_period])
            baseline_rpe = np.mean(np.array(rpe_values)[baseline_period])
            opioid_rpe = np.mean(np.array(rpe_values)[opioid_period])
            
            print(f"\n2. Opioid Effect Analysis:")
            print(f"   Baseline dopamine: {baseline_da:.3f}")
            print(f"   Opioid dopamine: {opioid_da:.3f}")
            print(f"   Dopamine increase: {((opioid_da - baseline_da) / baseline_da * 100):.1f}%")
            print(f"   Baseline RPE: {baseline_rpe:.3f}")
            print(f"   Opioid RPE: {opioid_rpe:.3f}")
            print(f"   RPE change: {((opioid_rpe - baseline_rpe) / abs(baseline_rpe) * 100):.1f}%")

def run_integrated_simulation():
    # Create and run integrated model
    integrated_model = IntegratedOpioidRewardModel()
    results = integrated_model.run_integrated_simulation()

if __name__ == "__main__":
    run_integrated_simulation() 