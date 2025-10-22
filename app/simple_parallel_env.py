#!/usr/bin/env python3
"""
Simple parallel environment wrapper using multiprocessing
No dependency on stable-baselines3
"""

import multiprocessing as mp
import numpy as np
from typing import List, Tuple, Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from simulation import FiveGEnvironment


class SimpleParallelEnv:
    """
    Simple parallel environment wrapper
    Chạy nhiều môi trường song song bằng multiprocessing
    
    Usage:
        envs = SimpleParallelEnv(n_envs=4, scenario='indoor_hotspot')
        obs = envs.reset()
        obs, rewards, dones, infos = envs.step(actions)
        envs.close()
    """
    
    def __init__(self, n_envs: int, scenario: str = 'indoor_hotspot', 
                 base_seed: int = 42, max_steps: int = None, scenarios_dir: str = None):
        self.n_envs = n_envs
        self.scenario = scenario
        self.base_seed = base_seed
        self.max_steps = max_steps
        self.scenarios_dir = scenarios_dir
        
        # Create pipes for communication with worker processes
        self.parent_conns, self.child_conns = zip(*[mp.Pipe() for _ in range(n_envs)])
        
        # Create worker processes
        self.processes = []
        for idx in range(n_envs):
            p = mp.Process(
                target=self._worker,
                args=(idx, self.child_conns[idx], scenario, base_seed + idx, max_steps, scenarios_dir)
            )
            p.daemon = True
            p.start()
            self.processes.append(p)
        
        # Get environment info from first worker
        self.parent_conns[0].send(('get_spaces', None))
        self.action_dim, self.obs_dim = self.parent_conns[0].recv()
        
        print(f"Created {n_envs} parallel environments")
        print(f"  Scenario: {scenario}")
        print(f"  Action dim: {self.action_dim}")
        print(f"  Obs dim: {self.obs_dim}")
    
    @staticmethod
    def _worker(idx: int, conn, scenario: str, seed: int, max_steps: int = None, scenarios_dir: str = None):
        """Worker process running a single environment"""
        # Create environment in worker process
        env = FiveGEnvironment(scenario=scenario, seed=seed, scenarios_dir=scenarios_dir)
        
        if max_steps is not None:
            env.sim_params.total_steps = max_steps
            env.sim_params.sim_time = max_steps * env.sim_params.time_step
        
        try:
            while True:
                cmd, data = conn.recv()
                
                if cmd == 'reset':
                    obs, info = env.reset(seed=seed)
                    conn.send((obs, info))
                
                elif cmd == 'step':
                    action = data
                    obs, reward, done, truncated, info = env.step(action)
                    conn.send((obs, reward, done, truncated, info))
                
                elif cmd == 'get_spaces':
                    conn.send((env.action_dim, env.state_dim))
                
                elif cmd == 'get_results':
                    results = env.get_results()
                    conn.send(results)
                
                elif cmd == 'close':
                    break
                
                else:
                    raise ValueError(f"Unknown command: {cmd}")
        
        except KeyboardInterrupt:
            pass
        finally:
            conn.close()
    
    def reset(self) -> np.ndarray:
        """Reset all environments"""
        for conn in self.parent_conns:
            conn.send(('reset', None))
        
        results = [conn.recv() for conn in self.parent_conns]
        obs = np.array([r[0] for r in results])
        return obs
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Step all environments
        
        Args:
            actions: Array of shape (n_envs, action_dim)
        
        Returns:
            obs: (n_envs, obs_dim)
            rewards: (n_envs,)
            dones: (n_envs,)
            infos: List of dicts
        """
        for conn, action in zip(self.parent_conns, actions):
            conn.send(('step', action))
        
        results = [conn.recv() for conn in self.parent_conns]
        
        obs = np.array([r[0] for r in results])
        rewards = np.array([r[1] for r in results])
        dones = np.array([r[2] for r in results])
        infos = [r[4] for r in results]
        
        return obs, rewards, dones, infos
    
    def get_results(self) -> List[Dict]:
        """Get final results from all environments"""
        for conn in self.parent_conns:
            conn.send(('get_results', None))
        
        return [conn.recv() for conn in self.parent_conns]
    
    def close(self):
        """Close all worker processes"""
        for conn in self.parent_conns:
            try:
                conn.send(('close', None))
            except:
                pass
        
        for p in self.processes:
            p.join(timeout=1)
            if p.is_alive():
                p.terminate()
        
        for conn in self.parent_conns:
            conn.close()


def example_usage():
    """Example: Using SimpleParallelEnv"""
    print("\n=== Example: SimpleParallelEnv ===\n")
    
    # Create 4 parallel environments
    envs = SimpleParallelEnv(n_envs=4, scenario='indoor_hotspot', max_steps=20)
    
    # Reset
    obs = envs.reset()
    print(f"\nReset done. Obs shape: {obs.shape}")
    
    # Run some steps
    n_steps = 10
    print(f"\nRunning {n_steps} steps...\n")
    
    total_rewards = np.zeros(envs.n_envs)
    
    for step in range(n_steps):
        # Random actions for all environments
        actions = np.random.uniform(0.5, 1.0, (envs.n_envs, envs.action_dim))
        
        # Step all environments in parallel
        obs, rewards, dones, infos = envs.step(actions)
        
        total_rewards += rewards
        
        if step == 0 or step == n_steps - 1:
            print(f"Step {step}:")
            for i in range(envs.n_envs):
                print(f"  Env {i}: reward={rewards[i]:.4f}, "
                      f"energy={infos[i].get('cumulative_energy', 0):.4f} kWh")
    
    print(f"\nTotal rewards: {total_rewards}")
    
    # Get final results
    results = envs.get_results()
    print(f"\nFinal results:")
    for i, r in enumerate(results):
        print(f"  Env {i}: energy={r['final_energy_consumption']:.4f} kWh, "
              f"drop_rate={r['final_drop_rate']:.2f}%")
    
    # Close
    envs.close()
    print("\n✓ Example completed!\n")


def benchmark_speedup():
    """Benchmark parallel vs sequential"""
    import time
    
    print("\n=== Benchmark: Parallel vs Sequential ===\n")
    
    n_envs = 4
    n_steps = 20
    scenario = 'indoor_hotspot'
    
    # 1. Sequential
    print("1. Sequential execution...")
    start = time.time()
    for i in range(n_envs):
        env = FiveGEnvironment(scenario, 42 + i)
        env.sim_params.total_steps = n_steps
        env.sim_params.sim_time = n_steps * env.sim_params.time_step
        obs, _ = env.reset()
        for _ in range(n_steps):
            action = np.random.uniform(0.5, 1.0, env.action_dim)
            env.step(action)
    sequential_time = time.time() - start
    print(f"   Time: {sequential_time:.2f}s\n")
    
    # 2. Parallel
    print("2. Parallel execution...")
    start = time.time()
    envs = SimpleParallelEnv(n_envs=n_envs, scenario=scenario, max_steps=n_steps)
    obs = envs.reset()
    for _ in range(n_steps):
        actions = np.random.uniform(0.5, 1.0, (n_envs, envs.action_dim))
        envs.step(actions)
    envs.close()
    parallel_time = time.time() - start
    print(f"   Time: {parallel_time:.2f}s\n")
    
    # Results
    speedup = sequential_time / parallel_time
    print(f"Results:")
    print(f"  Sequential: {sequential_time:.2f}s")
    print(f"  Parallel:   {parallel_time:.2f}s")
    print(f"  Speedup:    {speedup:.2f}x")
    print(f"  Efficiency: {speedup/n_envs*100:.1f}%\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['example', 'benchmark'], default='example')
    args = parser.parse_args()
    
    if args.mode == 'example':
        example_usage()
    else:
        benchmark_speedup()

