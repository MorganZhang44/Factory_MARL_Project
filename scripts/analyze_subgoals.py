import sys
from pathlib import Path
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent.parent))
from marl.envs.pursuit_env import PursuitEnv
from marl.policies.actor import Actor
from marl.utils.normalizer import RunningMeanStd

def main():
    CFG_PATH = "configs/mappo_config.yaml"
    CKPT_PATH = "results/checkpoints/final.pt"

    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    obs_norm = RunningMeanStd(shape=(int(cfg["env"].get("obs_dim", 13)),))
    obs_norm.mean = ckpt["obs_norm_mean"]
    obs_norm.var = ckpt["obs_norm_var"]
    obs_norm.count = ckpt["obs_norm_count"]
    
    MAP_HALF = float(cfg["env"]["map_half"])
    actor = Actor(int(cfg["env"].get("obs_dim", 13)), 2, 64, map_half=MAP_HALF)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    env = PursuitEnv(cfg, render_mode=None)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))     # ------------------------ CHANGE ------------------------
    axes = axes.flatten()

    for ax_idx in range(6): # ------------------------ CHANGE ------------------------
        ax = axes[ax_idx]
        env.reset(seed=42 + ax_idx)
        
        # Ensure intruder is moving to avoid OOD issues
        angle = env.np_random.uniform(0, 2 * np.pi)
        env.target_vel = np.array([np.cos(angle), np.sin(angle)]) * env.intruder_spd

        obs = env._get_obs()
        obs_n = obs_norm.normalize(obs)
        
        with torch.no_grad():
            act, _ = actor.get_action(torch.FloatTensor(obs_n), deterministic=True)
        act_np = act.numpy()
        
        print(f"--- Scenario {ax_idx+1} ---")
        print(f"Target Pos: {env.target_pos}")
        print(f"A1 Pos: {env.agent_pos[0]}, A1 Act (Offset): {act_np[0]}")
        print(f"A2 Pos: {env.agent_pos[1]}, A2 Act (Offset): {act_np[1]}")
        
        # Draw Map
        grid = env.obs_map.get_grid()
        ax.imshow(
            np.flipud(grid), cmap="gray_r", alpha=0.4,
            extent=[-MAP_HALF, MAP_HALF, -MAP_HALF, MAP_HALF]
        )

        colors = ["royalblue", "steelblue"]
        for i in range(env.n_agents):
            pos = env.agent_pos[i]
            
            # Preserve direction but limit magnitude to e.g., 2.0 meters
            raw_offset = act_np[i]
            mag = float(np.linalg.norm(raw_offset))
            max_mag = 1.5
            if mag > max_mag:
                offset = (raw_offset / mag) * max_mag
            else:
                offset = raw_offset
                
            sg = np.clip(pos + offset, -MAP_HALF + 0.3, MAP_HALF - 0.3)
            
            # Plot agent
            ax.plot(pos[0], pos[1], 'o', color=colors[i], markersize=10, label=f"Agent {i+1}")
            
            # Plot subgoal
            ax.plot(sg[0], sg[1], 'X', color='orange', markersize=8)
            
            # Draw line from agent to subgoal
            ax.arrow(pos[0], pos[1], sg[0]-pos[0], sg[1]-pos[1], 
                     head_width=0.3, head_length=0.4, fc='orange', ec='orange', 
                     linestyle='--', length_includes_head=True, alpha=0.7)
            
            # Plot A* path if we want to
            from marl.utils.astar import astar
            path = astar(env.obs_map, tuple(pos.tolist()), tuple(sg.tolist()))
            if path:
                px, py = zip(*path)
                ax.plot(px, py, ":", color=colors[i], alpha=0.6, lw=2)

        # Plot Intruder
        t_pos = env.target_pos
        t_vel = env.target_vel
        ax.plot(t_pos[0], t_pos[1], 'r*', markersize=14, label="Intruder")
        
        # Intruder velocity arrow
        ax.arrow(t_pos[0], t_pos[1], t_vel[0]*2, t_vel[1]*2, 
                 head_width=0.3, head_length=0.4, fc='red', ec='red', 
                 alpha=0.5, length_includes_head=True)

        ax.set_xlim(-MAP_HALF, MAP_HALF)
        ax.set_ylim(-MAP_HALF, MAP_HALF)
        ax.set_aspect("equal")
        ax.set_title(f"Random Scenario {ax_idx+1}")
        
        if ax_idx == 0:
            ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig("subgoal_analysis.png", dpi=150)
    print("Saved subgoal_analysis.png")

if __name__ == "__main__":
    main()
