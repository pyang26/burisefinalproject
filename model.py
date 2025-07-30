#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple
import random

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
        # Processing history - track all components
        self.actual_rewards = []
        self.expected_rewards = []
        self.rpe_history = []
        self.dopamine_history = []
        
    def generate_dopamine_scenario(self, scenario: DopamineScenario, step: int) -> DopamineInput:
        if scenario == DopamineScenario.HIGH_DOPAMINE:
            level = 0.8 + 0.2 * np.sin(step * 0.1)
        elif scenario == DopamineScenario.LOW_DOPAMINE:
            level = 0.2 + 0.1 * np.sin(step * 0.1)
        elif scenario == DopamineScenario.NORMAL_DOPAMINE:
            level = 0.5 + 0.1 * np.sin(step * 0.1)
        elif scenario == DopamineScenario.VARIABLE_DOPAMINE:
            level = 0.3 + 0.4 * np.sin(step * 0.3)
        elif scenario == DopamineScenario.DEPLETED_DOPAMINE:
            level = 0.1 + 0.05 * np.sin(step * 0.1)
        else:
            level = 0.5
        return DopamineInput(level=level, scenario=scenario, timestamp=step)
    
    def generate_rewards_from_dopamine(self, dopamine_input: DopamineInput, step: int) -> RewardData:
        # Generate actual reward from environment (external) with slight oscillation
        # Expected rewards are influenced by dopamine (internal predictions)
        
        # Generate actual reward from environment - with slight oscillation
        base_reward = 0.5
        oscillation = 0.05 * np.sin(step * 0.2)  # Small oscillation
        actual_reward = base_reward + oscillation
        
        # Generate expected reward based on dopamine level (dopamine affects predictions)
        if dopamine_input.level > 0.7:  # High dopamine → very optimistic predictions
            expected_reward = actual_reward * 4.0 + 3.0 + dopamine_input.level * 4.0
        elif dopamine_input.level > 0.4:  # Medium dopamine → realistic predictions
            expected_reward = actual_reward * 2.7 + 1.0 + dopamine_input.level * 2.5
        elif dopamine_input.level > 0.2:  # Low dopamine → pessimistic predictions
            expected_reward = actual_reward * 0.8 - 1.5 + dopamine_input.level * 1.0  
        else:  # Very low dopamine → very pessimistic predictions
            expected_reward = actual_reward * 0.2 - 3.0 + dopamine_input.level * 0.5 
        
        return RewardData(
            actual_reward=actual_reward,
            expected_reward=expected_reward,
            timestamp=step
        )
    
    def process_dopamine_step(self, dopamine_input: DopamineInput) -> Dict:
        # Generate rewards and expectations from dopamine input
        reward_data = self.generate_rewards_from_dopamine(dopamine_input, dopamine_input.timestamp)
        # Compute RPE using dopamine and reward
        rpe, expected_reward = self.rpe_computer.compute_rpe(dopamine_input, reward_data)
        
        # Record history
        self.actual_rewards.append(reward_data.actual_reward)
        self.expected_rewards.append(expected_reward)
        self.rpe_history.append(rpe)
        self.dopamine_history.append(dopamine_input.level)
        
        return {
            'actual_reward': reward_data.actual_reward,
            'expected_reward': expected_reward,
            'rpe': rpe,
            'dopamine_level': dopamine_input.level
        }
    
    def run_dopamine_experiment(self, scenario: DopamineScenario, num_steps: int = 100) -> Dict:
        experiment_results = []
        
        for step in range(num_steps):
            # Generate dopamine input (the ONLY input)
            dopamine_input = self.generate_dopamine_scenario(scenario, step)
            # Process step with dopamine input
            result = self.process_dopamine_step(dopamine_input)
            experiment_results.append(result)
            
            if step % 20 == 0:
                print(f"Step {step}: DA={dopamine_input.level:.3f}, "
                      f"Actual={result['actual_reward']:.3f}, "
                      f"Expected={result['expected_reward']:.3f}, "
                      f"RPE={result['rpe']:.3f}")
        return experiment_results
    
    def plot_dopamine_correlation_analysis(self, scenario: DopamineScenario):
        if not self.rpe_history:
            print("No data to plot.")
            return
        
        fig = plt.figure(figsize=(16, 10))
        
        # Group 1: Dopamine-driven components (2x3)
        ax1 = plt.subplot(2, 3, 1)
        ax2 = plt.subplot(2, 3, 2)
        ax3 = plt.subplot(2, 3, 3)
        ax4 = plt.subplot(2, 3, 4)
        ax5 = plt.subplot(2, 3, 5)
        ax6 = plt.subplot(2, 3, 6)
        
        steps = range(len(self.rpe_history))
        
        # 1. Dopamine input over time (THE DRIVER)
        ax1.plot(steps, self.dopamine_history, 'purple', linewidth=3)
        ax1.set_title('Dopamine Input (THE DRIVER)', fontweight='bold')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Dopamine Level')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # 2. Rewards and expectations
        ax2.plot(steps, self.actual_rewards, 'g-', linewidth=2, label='Actual Reward (Environment)')
        ax2.plot(steps, self.expected_rewards, 'b--', linewidth=2, label='Expected Reward (Dopamine)')
        ax2.set_title('Actual vs Expected Rewards', fontweight='bold')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Reward Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # 3. RPE over time
        ax3.plot(steps, self.rpe_history, 'r-', linewidth=2)
        ax3.set_title('RPE Over Time', fontweight='bold')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('RPE')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # 4. KEY: Dopamine vs RPE correlation
        ax4.scatter(self.dopamine_history, self.rpe_history, alpha=0.6, s=30, color='orange')
        ax4.set_title('Dopamine Input vs RPE Output', fontweight='bold')
        ax4.set_xlabel('Dopamine Level')
        ax4.set_ylabel('RPE')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # 5. Dopamine vs Actual Reward correlation
        ax5.scatter(self.dopamine_history, self.actual_rewards, alpha=0.6, s=30, color='green')
        ax5.set_title('Dopamine vs Actual Reward', fontweight='bold')
        ax5.set_xlabel('Dopamine Level')
        ax5.set_ylabel('Actual Reward')
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # 6. Temporal alignment: Dopamine → Rewards → RPE
        ax_twin = ax6.twinx()
        
        # Plot dopamine on primary axis
        line1 = ax6.plot(steps, self.dopamine_history, 'purple', linewidth=3, label='Dopamine Input')
        ax6.set_xlabel('Time Step')
        ax6.set_ylabel('Dopamine Level', color='purple')
        ax6.tick_params(axis='y', labelcolor='purple')
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim(0, 1)
        
        # Plot RPE on secondary axis
        line2 = ax_twin.plot(steps, self.rpe_history, 'r-', linewidth=2, label='RPE Output')
        ax_twin.set_ylabel('RPE', color='r')
        ax_twin.tick_params(axis='y', labelcolor='r')
        ax_twin.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        ax6.set_title('Dopamine Input → RPE Output', fontweight='bold')
        
        # Add legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax6.legend(lines, labels, loc='upper right')
        
        plt.suptitle(f'Dopamine-Centered Reward Processing: {scenario.value.upper()}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Print focused analysis
        self._print_dopamine_correlation_analysis(scenario)
    
    def _print_dopamine_correlation_analysis(self, scenario: DopamineScenario):
        """Print analysis focusing on dopamine → RPE correlation."""
        print("\n" + "="*60)
        print(f"DOPAMINE → RPE CORRELATION ANALYSIS")
        print(f"Scenario: {scenario.value.upper()}")
        print("="*60)
        
        print(f"\n1. Overall Statistics:")
        print(f"   Total steps: {len(self.rpe_history)}")
        print(f"   Average dopamine level: {np.mean(self.dopamine_history):.3f}")
        print(f"   Average actual reward: {np.mean(self.actual_rewards):.3f}")
        print(f"   Average expected reward: {np.mean(self.expected_rewards):.3f}")
        print(f"   Average RPE: {np.mean(self.rpe_history):.3f}")
        
        print(f"\n2. KEY: Dopamine-RPE Correlation:")
        dopamine_rpe_corr = np.corrcoef(self.dopamine_history, self.rpe_history)[0, 1]
        print(f"   Dopamine ↔ RPE correlation: {dopamine_rpe_corr:.3f}")
        
        if dopamine_rpe_corr > 0.3:
            print("✅ Strong positive correlation: Higher dopamine → Higher RPE")
        elif dopamine_rpe_corr > 0.1:
            print("⚠️  Moderate correlation: Some dopamine-RPE relationship")
        else:
            print("❌ Weak correlation: Limited dopamine-RPE relationship")
            
        print(f"\n3. Dopamine-Reward Correlation:")
        dopamine_reward_corr = np.corrcoef(self.dopamine_history, self.actual_rewards)[0, 1]
        print(f"   Dopamine ↔ Actual reward correlation: {dopamine_reward_corr:.3f}")

        print(f"\n4. RPE Analysis:")
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
        results = processor.run_dopamine_experiment(scenario, num_steps=100)
        # Plot dopamine correlation analysis
        processor.plot_dopamine_correlation_analysis(scenario)
    print("\n" + "="*60)
    print("Dopamine-centered experiments complete!")
    print("="*60)

if __name__ == "__main__":
    run_dopamine_centered_experiments() 