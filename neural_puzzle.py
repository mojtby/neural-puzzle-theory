import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class NeuralPuzzle:
    """Neural Puzzle Theory simulation with full molecular mechanisms"""
    
    def __init__(self, N_neurons=10000):
        # Parameters (from Appendix A)
        self.V0 = -70.0  # mV
        self.tau_LT = 20 * 365 * 24 * 3600  # years to seconds
        self.lmbda = 0.5  # mV/nm
        self.tau_Ca = 0.1  # s
        self.kappa = 0.1  # uM per spike
        self.tau_MT = 24 * 3600  # s
        self.alpha = 0.3  # nm
        
        # Channel mechanism parameters (Section 4.5)
        self.gamma_MAP = 0.005  # channels/s per MAP site
        self.gamma_kinesin = 0.01  # channels/s per binding event
        self.Phi_max = 100  # max binding sites
        self.K_d = 0.5  # nm
        self.n_Hill = 2
        self.P0 = 0.8  # baseline kinesin binding
        self.u0 = 1.0  # nm (deformation scale for kinesin)
        self.tau_deg = 7 * 24 * 3600  # channel degradation time (7 days)
        
        # Quantum extension parameters (Section 11)
        self.Delta = 0.1  # meV (tunneling)
        self.dipole = 1000  # Debye
        self.tau_coherence = 10e-6  # 10 microseconds
        
        # State variables
        self.V_rest = self.V0 + 15 * np.random.randn(N_neurons)
        self.V_rest = np.clip(self.V_rest, -85, -55)
        self.Ca = np.zeros(N_neurons)
        self.u_MT = np.zeros(N_neurons)
        self.nK = 1000 + 200 * np.random.randn(N_neurons)  # baseline channels
        self.quantum_state = np.zeros(N_neurons, dtype=complex)  # for quantum extension
        
    def wake_step(self, S, dt=0.001):
        """Single step during wakefulness"""
        # Action potential frequency
        f = 10 + 0.5 * (self.V_rest - self.V0) + 0.1 * S + 0.1 * np.random.randn()
        
        # Calcium dynamics
        dCa = (self.kappa * f - self.Ca / self.tau_Ca) * dt
        self.Ca += dCa
        
        return f
    
    def mechanism_A_allosteric(self, u):
        """MAP-mediated allosteric regulation"""
        return self.Phi_max * (u**self.n_Hill) / (self.K_d**self.n_Hill + u**self.n_Hill)
    
    def mechanism_B_kinesin(self, u):
        """Kinesin-mediated trafficking"""
        return self.P0 * np.exp(-np.abs(u) / self.u0)
    
    def mechanism_C_scaffold(self, u, nK):
        """Scaffold protein reorganization"""
        k_on_scaff = 0.001
        k_off_scaff = 0.0005
        return k_on_scaff * (2000 - nK) - k_off_scaff * nK * np.exp(-np.abs(u) / self.u0)
    
    def sleep_step(self, dt=3600):
        """Single hour of sleep with full molecular mechanisms"""
        # MT deformation with optional quantum coherence
        du = (self.alpha * np.tanh(self.Ca) - self.u_MT / self.tau_MT) * dt
        
        # Add quantum coherence if enabled (simplified)
        if hasattr(self, 'quantum_enabled') and self.quantum_enabled:
            # Coherent oscillation term
            omega = 2 * np.pi * 50e9  # 50 GHz
            du += 0.01 * self.alpha * np.sin(omega * dt) * dt
        
        self.u_MT += du
        
        # Channel redistribution with three mechanisms
        for i in range(len(self.u_MT)):
            u = np.abs(self.u_MT[i])
            
            # Mechanism A: Allosteric
            dn_A = self.gamma_MAP * self.mechanism_A_allosteric(u)
            
            # Mechanism B: Kinesin
            dn_B = self.gamma_kinesin * self.mechanism_B_kinesin(u)
            
            # Mechanism C: Scaffold
            dn_C = self.mechanism_C_scaffold(u, self.nK[i])
            
            # Degradation
            dn_deg = -self.nK[i] / self.tau_deg * dt
            
            # Update channel count
            self.nK[i] += (dn_A + dn_B + dn_C + dn_deg) * dt
            self.nK[i] = np.clip(self.nK[i], 500, 2000)  # physiological range
        
        # Resting potential update
        self.V_rest += self.lmbda * np.abs(self.u_MT) * (1 - np.exp(-dt/7200))
        
        return self.u_MT
    
    def quantum_evolution(self, dt):
        """Simplified quantum evolution (Section 11)"""
        # Heisenberg equation for <sigma_z>
        # d<sigma_z>/dt = - (<sigma_z> - <sigma_z>_0)/tau_coherence + driving
        target = np.tanh(self.Ca)  # tension-dependent target
        dsigma = -(self.quantum_state.real - target) / self.tau_coherence * dt
        self.quantum_state += dsigma
        return self.quantum_state.real
    
    def recall(self, cue_strength=0.1, T=1.0, dt=0.001):
        """Simulate memory recall"""
        V = self.V_rest + cue_strength * np.random.randn(len(self.V_rest))
        tau_m = 0.01  # s
        
        for t in np.arange(0, T, dt):
            dV = -(V - self.V_rest) / tau_m * dt
            V += dV
            
        return V
    
# Example simulation
if __name__ == "__main__":
    model = NeuralPuzzle(N_neurons=1000)

    # Wake period (8 hours)
    print("Wake period...")
    for hour in range(8):
        for minute in range(60):
            S = 1.0 if np.random.rand() > 0.5 else 0.0
            model.wake_step(S, dt=0.001)

    print(f"After wake: V_rest = {model.V_rest.mean():.1f} +/- {model.V_rest.std():.1f} mV")

    # Sleep period (8 hours)
    print("Sleep period...")
    for hour in range(8):
        model.sleep_step(dt=3600)

    print(f"After sleep: V_rest = {model.V_rest.mean():.1f} +/- {model.V_rest.std():.1f} mV")

    # Test recall
    recalled = model.recall(cue_strength=0.05)
    print(f"Recall correlation: {np.corrcoef(model.V_rest, recalled)[0,1]:.3f}")