#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple

class DopamineScenario(Enum):
    #different simulated dopamine scenarios
    HIGH_DOPAMINE = "high_dopamine"
    LOW_DOPAMINE = "low_dopamine"
    NORMAL_DOPAMINE = "normal_dopamine"
    VARIABLE_DOPAMINE = "variable_dopamine"
    DEPLETED_DOPAMINE = "depleted_dopamine"

@dataclass
class DopamineInput:
    level: float  # Current dopamine level (0.0 to 1.0)
    scenario: DopamineScenario
    timestamp: int

@dataclass
#not sure if this is needed for now
class RewardData:
    actual_reward: float
    expected_reward: float
    timestamp: int

class DopamineCenteredRPE(nn.Module):
    def __init__(self, state_dim: int = 8):
        super().__init__()
        self.state_dim = state_dim
        # Expectation network (predicts expected rewards based on dopamine)
        self.expectation_network = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        # Dopamine modulation network
        self.dopamine_modulator = nn.Sequential(
            nn.Linear(1, 16),  # Dopamine level input
            nn.ReLU(),
            nn.Linear(16, 8), 
            nn.ReLU(),
            nn.Linear(8, 1) 
        )
        
    def compute_rpe(self, dopamine_input: DopamineInput, reward_data: RewardData) -> Tuple[float, float]:
        # Expected reward based on dopamine (learned function)
        dopamine_tensor = torch.tensor([[dopamine_input.level]], dtype=torch.float32)
        expected_reward = self.expectation_network(dopamine_tensor).item()
        # RPE: actual reward - expected reward
        rpe = reward_data.actual_reward - expected_reward
        return rpe, expected_reward

class DopamineCenteredProcessor:
    def __init__(self, state_dim: int = 8):
        self.rpe_computer = DopamineCenteredRPE(state_dim)
        self.state_dim = state_dim
        
        # Temporal difference learning parameters
        self.gamma = 0.9  # Discount factor
        self.eta = 0.1    # Learning rate
        
        # State value tracking
        self.current_state_value = 0.0
        self.next_state_value = 0.0
        
        # Processing history - track all components
        self.actual_rewards = []
        self.expected_rewards = []
        self.rpe_history = []
        self.dopamine_history = []
        self.state_values = []
        
    def generate_dopamine_scenario(self, scenario: DopamineScenario, step: int) -> DopamineInput:
        # More continuous time progression with smaller increments
        time_factor = step * 0.02  
        if scenario == DopamineScenario.HIGH_DOPAMINE:
            level = 0.8 + 0.2 * np.sin(time_factor)
        elif scenario == DopamineScenario.LOW_DOPAMINE:
            level = 0.2 + 0.1 * np.sin(time_factor)
        elif scenario == DopamineScenario.NORMAL_DOPAMINE:
            level = 0.5 + 0.1 * np.sin(time_factor)
        elif scenario == DopamineScenario.VARIABLE_DOPAMINE:
            level = 0.3 + 0.4 * np.sin(time_factor * 1.5)
        elif scenario == DopamineScenario.DEPLETED_DOPAMINE:
            level = 0.1 + 0.05 * np.sin(time_factor)
        else:
            level = 0.5
        return DopamineInput(level=level, scenario=scenario, timestamp=step)
    
    def generate_rewards_from_dopamine(self, dopamine_input: DopamineInput, step: int, rpe_history: List[float] = None) -> RewardData:
        # Generate actual reward from environment (external) with slight oscillation
        # Expected rewards are influenced by dopamine (internal predictions) and previous RPEs
        
        if step <= 250:
            actual_reward = 0.5
        else:
            actual_reward = 0.3
        
        # Enhanced dopamine-expected reward relationship
        # Based on sigmoidal response curve from dopamine to expected reward
        # E(R) = R_max * (DA^n / (K^n + DA^n)) where DA = dopamine level
        
        R_max = actual_reward * 4.0  # Increased maximum expected reward
        K = 0.3  # Lower half-saturation constant (more sensitive to dopamine)
        n = 3.0  # Higher Hill coefficient (steeper response)
        
        # Enhanced sigmoidal dopamine response: E(R) = R_max * (DA^n / (K^n + DA^n))
        dopamine_response = (dopamine_input.level ** n) / (K ** n + dopamine_input.level ** n)
        base_expected = R_max * dopamine_response
        
        # Temporal difference RPE calculation with dopamine modulation
        # δt = rt + γV(St+1) - V(St)
        
        # Calculate temporal difference RPE
        td_rpe = actual_reward + self.gamma * self.next_state_value - self.current_state_value
        
        # Dopamine modulates the RPE calculation
        # Higher dopamine → enhanced RPE sensitivity
        # Lower dopamine → reduced RPE sensitivity
        dopamine_modulation = 1.0 + dopamine_input.level * 0.5  # 1.0 to 1.5 range
        modulated_td_rpe = td_rpe * dopamine_modulation
        
        # Update state value using temporal difference learning
        # V(St)new = V(St)old + η * δt
        self.current_state_value = self.current_state_value + self.eta * modulated_td_rpe
        
        # Use temporal difference RPE for expected reward adjustment
        rpe_adjustment = modulated_td_rpe * 0.3  # Scale down for stability
        
        # Final expected reward: sigmoidal dopamine response + TD RPE learning
        expected_reward = base_expected + rpe_adjustment 
        
        return RewardData(
            actual_reward=actual_reward,
            expected_reward=expected_reward,
            timestamp=step
        )
    
    def process_dopamine_step(self, dopamine_input: DopamineInput) -> Dict:
        # Generate rewards and expectations from dopamine input
        reward_data = self.generate_rewards_from_dopamine(dopamine_input, dopamine_input.timestamp, self.rpe_history)
        
        # Update next state value (simulate environment transition)
        self.next_state_value = self.current_state_value + np.random.normal(0, 0.1)  # Small random change
        
        # Compute temporal difference RPE with dopamine modulation
        td_rpe = reward_data.actual_reward + self.gamma * self.next_state_value - self.current_state_value
        dopamine_modulation = 1.0 + dopamine_input.level * 0.5
        modulated_td_rpe = td_rpe * dopamine_modulation
        
        # Update state value using temporal difference learning
        self.current_state_value = self.current_state_value + self.eta * modulated_td_rpe
        
        # Use temporal difference RPE as the main RPE
        rpe = modulated_td_rpe
        expected_reward = reward_data.expected_reward
        
        # Record history
        self.actual_rewards.append(reward_data.actual_reward)
        self.expected_rewards.append(expected_reward)
        self.rpe_history.append(rpe)
        self.dopamine_history.append(dopamine_input.level)
        self.state_values.append(self.current_state_value)
        
        return {
            'actual_reward': reward_data.actual_reward,
            'expected_reward': expected_reward,
            'rpe': rpe,
            'dopamine_level': dopamine_input.level
        }
    
    def run_dopamine_experiment(self, scenario: DopamineScenario, num_steps: int = 500) -> Dict:
        experiment_results = []
        
        for step in range(num_steps):
            # Generate dopamine input (the ONLY input)
            dopamine_input = self.generate_dopamine_scenario(scenario, step)
            # Process step with dopamine input
            result = self.process_dopamine_step(dopamine_input)
            experiment_results.append(result)
            
            if step % 100 == 0:
                print(f"Step {step}: DA={dopamine_input.level:.3f}, "
                      f"Actual={result['actual_reward']:.3f}, "
                      f"Expected={result['expected_reward']:.3f}, "
                      f"RPE={result['rpe']:.3f}")
        return experiment_results
    
    def plot_dopamine_correlation_analysis(self, scenario: DopamineScenario):
        if not self.rpe_history:
            print("No data to plot.")
            return
        
        fig = plt.figure(figsize=(12, 8))
        
        # Group 1: Dopamine-driven components (2x2)
        ax1 = plt.subplot(2, 2, 1)
        ax2 = plt.subplot(2, 2, 2)
        ax3 = plt.subplot(2, 2, 3)
        ax4 = plt.subplot(2, 2, 4)
        
        steps = range(len(self.rpe_history))
        
        # 1. Dopamine input over time
        ax1.plot(steps, self.dopamine_history, 'purple', linewidth=3)
        ax1.set_title('Dopamine Input', fontweight='bold')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Dopamine Level (arbitrary units)')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # 2. Rewards and expectations
        ax2.plot(steps, self.actual_rewards, 'g-', linewidth=2, label='Actual Reward (Environment)')
        ax2.plot(steps, self.expected_rewards, 'b--', linewidth=2, label='Expected Reward (Dopamine)')
        ax2.set_title('Actual vs Expected Rewards', fontweight='bold')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Reward Value (arbitrary units)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # 3. RPE over time
        ax3.plot(steps, self.rpe_history, 'r-', linewidth=2)
        ax3.set_title('RPE Over Time', fontweight='bold')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('RPE (arbitrary units)')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # 4. Temporal alignment: Dopamine → Rewards → RPE
        ax_twin = ax4.twinx()
        
        # Plot dopamine on primary axis
        line1 = ax4.plot(steps, self.dopamine_history, 'purple', linewidth=3, label='Dopamine Input')
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Dopamine Level (arbitrary units)', color='purple')
        ax4.tick_params(axis='y', labelcolor='purple')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
        
        # Plot RPE on secondary axis
        line2 = ax_twin.plot(steps, self.rpe_history, 'r-', linewidth=2, label='RPE Output')
        ax_twin.set_ylabel('RPE', color='r')
        ax_twin.tick_params(axis='y', labelcolor='r')
        ax_twin.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        ax4.set_title('Dopamine Input → RPE Output', fontweight='bold')
        
        # Add legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper right')
        
        plt.suptitle(f'Dopamine-Centered Reward Processing: {scenario.value.upper()}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        self._print_dopamine_correlation_analysis(scenario)
    
    def _print_dopamine_correlation_analysis(self, scenario: DopamineScenario):
        """Print analysis focusing on dopamine → RPE correlation."""
        print("\n" + "="*60)
        print(f"DOPAMINE → RPE CORRELATION ANALYSIS")
        print(f"Scenario: {scenario.value.upper()}")
        print("="*60)
        
        print(f"Overall Statistics:")
        print(f"   Total steps: {len(self.rpe_history)}")
        print(f"   Average dopamine level: {np.mean(self.dopamine_history):.3f}")
        print(f"   Average actual reward: {np.mean(self.actual_rewards):.3f}")
        print(f"   Average expected reward: {np.mean(self.expected_rewards):.3f}")
        print(f"   Average RPE: {np.mean(self.rpe_history):.3f}")

        print(f"RPE Analysis:")
        positive_rpe = [rpe for rpe in self.rpe_history if rpe > 0]
        negative_rpe = [rpe for rpe in self.rpe_history if rpe < 0]
        print(f"   Positive RPE events: {len(positive_rpe)}")
        print(f"   Negative RPE events: {len(negative_rpe)}")
        print(f"   Average positive RPE: {np.mean(positive_rpe) if positive_rpe else 0:.3f}")
        print(f"   Average negative RPE: {np.mean(negative_rpe) if negative_rpe else 0:.3f}")

def run_dopamine_centered_experiments():
    print("="*60)
    print("DOPAMINE-CENTERED REWARD PROCESSING")
    print("="*60)
    
    # Define dopamine scenarios
    scenarios = [
        DopamineScenario.HIGH_DOPAMINE,
        DopamineScenario.LOW_DOPAMINE,
        DopamineScenario.NORMAL_DOPAMINE,
        DopamineScenario.VARIABLE_DOPAMINE,
        DopamineScenario.DEPLETED_DOPAMINE
    ]
    
    for scenario in scenarios:
        # Create new processor for each scenario
        processor = DopamineCenteredProcessor(state_dim=8)
        # Run experiment with dopamine as ONLY input
        results = processor.run_dopamine_experiment(scenario, num_steps=500)
        # Plot dopamine correlation analysis
        processor.plot_dopamine_correlation_analysis(scenario)

if __name__ == "__main__":
    run_dopamine_centered_experiments() 