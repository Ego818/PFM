from __future__ import annotations
import os, sys, time, json, argparse
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer

sys.path.insert(0, ".")
from marl_exploration_3d import MARLExploration3D, ExplorationConfig
from entrenar_agentes import TEAMS, CURRICULUM, ARCHETYPES, AgentObsWrapper

OUT_DIR = "checkpoints_sb3"
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE HIPERPARÁMETROS RECOMENDADA
# ═══════════════════════════════════════════════════════════════
def _ppo_kwargs(max_steps: int, n_agents: int) -> dict:
    # n_steps acotado: no más de 512 para que haya múltiples episodios por rollout
    # y la señal GAE no sea de un solo episodio monolítico
    n_steps = min(max_steps, 512)
    rollout = n_steps * n_agents
    batch   = max(64, rollout // 8)

    return dict(
        learning_rate  = 3e-4,
        n_steps        = n_steps,
        batch_size     = batch,
        n_epochs       = 8,                 # más épocas → mejor uso de cada rollout
        gamma          = 0.995,
        gae_lambda     = 0.95,
        clip_range     = 0.2,
        ent_coef       = 0.05,              # AUMENTADO: evita colapso de política en sparse reward
        vf_coef        = 0.5,
        max_grad_norm  = 0.5,
        policy_kwargs  = dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose        = 1,
        device         = "auto",
    )


# ═══════════════════════════════════════════════════════════════
# 1. WRAPPER MARL → VecEnv REPARADO (Garantiza flujo de Gradiente)
# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════
# 1. WRAPPER MARL → VecEnv COMPLETAMENTE OPERATIVO
# ═══════════════════════════════════════════════════════════════
class SB3MultiAgentVecEnv(VecEnv):
    def __init__(self, env: MARLExploration3D, archetype_id: int):
        self.env           = env
        self.arch          = ARCHETYPES[archetype_id]
        self.n_agents      = env.cfg.n_agents
        self.agent_indices = list(range(self.n_agents))
        self.info: Dict    = {}

        # Medir dimensión real haciendo un reset
        obs_p, _ = self.env.reset()
        obs_dim  = obs_p[0].shape   # shape del vector real que genera el entorno

        observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_dim, dtype=np.float32
        )
        action_space = gym.spaces.Discrete(self.env.n_actions)

        self.metadata    = {"render_modes": ["rgb_array"]}
        self.render_mode = "rgb_array"

        super().__init__(
            num_envs=self.n_agents,
            observation_space=observation_space,
            action_space=action_space,
        )
        self.actions_dict: Dict[int, int] = {}

    def reset(self, **kwargs) -> np.ndarray:
        seed = kwargs.get("seed", None)
        obs_dict, self.info = (
            self.env.reset(seed=seed) if seed is not None else self.env.reset()
        )
        return self._get_obs(obs_dict)

    def _get_obs(self, obs_dict) -> np.ndarray:
        return np.array(
            [obs_dict[i] for i in self.agent_indices], dtype=np.float32
        )

    def step_async(self, actions: np.ndarray) -> None:
        self.actions_dict = {i: int(act) for i, act in enumerate(actions)}

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        # Silenciar prints internos del entorno
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            obs_next, rewards, terminated, truncated, info = self.env.step(self.actions_dict)
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout

        self.info = info
        done      = terminated or truncated
        dones     = np.array([done] * self.n_agents, dtype=bool)
        rews_arr  = np.array([rewards[i] for i in self.agent_indices], dtype=np.float32)
        obs_arr   = self._get_obs(obs_next)
        infos     = [info.copy() for _ in range(self.n_agents)]

        if done:
            for idx in range(self.n_agents):
                infos[idx]["terminal_observation"]  = obs_arr[idx]
                infos[idx]["TimeLimit.truncated"]    = truncated
            obs_arr = self.reset()

        return obs_arr, rews_arr, dones, infos

    # ── Métodos abstractos de VecEnv ────────────────────────────

    def close(self) -> None:
        self.env.close()

    def get_attr(self, attr_name: str, indices=None) -> List[Any]:
        val = getattr(self, attr_name, None)
        return [val] * self.num_envs

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        setattr(self, attr_name, value)

    def env_method(self, method_name: str, *args, indices=None, **kwargs) -> List[Any]:
        return [None] * self.num_envs

    def env_is_wrapped(self, wrapper_class: type, indices=None) -> List[bool]:
        return [False] * self.num_envs

# ═══════════════════════════════════════════════════════════════
# 2. CALLBACK DE SEGUIMIENTO
# ═══════════════════════════════════════════════════════════════
class MARLCurriculumCallback(BaseCallback):
    def __init__(self, advance_cov: float, advance_k: int, verbose=0):
        super().__init__(verbose)
        self.advance_cov = advance_cov
        self.advance_k = advance_k
        self.recent_coverages = []
        # Acumulador interno para calcular la recompensa real del episodio
        self.episode_rewards_accumulator = 0.0

    def _on_step(self) -> bool:
        # 1. Acumular de forma continua las recompensas que reciben todos los agentes en este paso
        # 'rewards' es un array con las recompensas del paso actual de cada sub-entorno
        if "rewards" in self.locals:
            self.episode_rewards_accumulator += np.sum(self.locals["rewards"])

        # 2. Verificar de forma estricta si el episodio ha terminado por completo (Auto-Reset)
        if self.locals["dones"].any(): 
            # Extraer información de cobertura de manera segura desde el VecEnv de SB3
            info_attr = self.training_env.get_attr("info")
            if isinstance(info_attr, list) and len(info_attr) > 0:
                info = info_attr[0]
            else:
                info = info_attr if isinstance(info_attr, dict) else self.training_env.info
                
            cov = info.get("coverage_ratio", 0.0)
            self.recent_coverages.append(cov)
            
            if len(self.recent_coverages) > self.advance_k:
                self.recent_coverages.pop(0)
                
            mean_cov = np.mean(self.recent_coverages)
            
            # 🚀 IMPRESIÓN ÚNICA POR EPISODIO: Muestra Cobertura y Recompensa total acumulada
            print(f"🟩 [FIN EPISODIO] Cobertura: {cov*100:.2f}% | Media móvil: {mean_cov*100:.2f}% | Recompensa Total Equipo: {self.episode_rewards_accumulator:.2f}")
            
            # Resetear el acumulador para el inicio del siguiente episodio independiente
            self.episode_rewards_accumulator = 0.0
                
            # Evaluar condición de Curriculum para avanzar de Fase
            if len(self.recent_coverages) >= self.advance_k and mean_cov >= self.advance_cov:
                print(f"│ ⚡ Cambio de fase autorizado (Media: {mean_cov*100:.2f}% >= Objetivo: {self.advance_cov*100:.2f}%)")
                return False
                
        return True



# ═══════════════════════════════════════════════════════════════
# 3. PROCESO PRINCIPAL
# ═══════════════════════════════════════════════════════════════
def train_team_sb3(team_id: int, start_phase: int = 1, is_fast: bool = False):
    team = TEAMS[team_id]
    # Usa el arquetipo dominante del equipo (mayor complejidad de comunicación)
    dominant_arch_id = team.dominant.id

    model = None

    for phase in CURRICULUM:
        # ── RESPETAR start_phase ──────────────────────────────
        if phase.idx < start_phase or phase.n_episodes == 0:
            print(f"  [SKIP] Fase {phase.idx}: {phase.name}")
            continue

        if is_fast:
            phase.grid_h      = min(phase.grid_h, 25)
            phase.grid_w      = min(phase.grid_w, 25)
            phase.max_steps   = min(phase.max_steps, 300)
            phase.n_episodes  = max(5, phase.n_episodes // 30)

        print(f"\n{'='*70}")
        print(f"🚀 FASE {phase.idx}: {phase.name}  |  Mapa: {phase.grid_h}×{phase.grid_w}×{phase.n_floors}")
        print(f"{'='*70}")
        
        rwall = 0.0 if phase.idx <= 1 else -0.01
        cfg = ExplorationConfig(
            grid_h        = phase.grid_h,
            grid_w        = phase.grid_w,
            n_floors      = phase.n_floors,
            wall_density  = phase.wall_density,
            n_stairs      = phase.n_stairs,
            min_connected = min(
                phase.min_connected,
                int(phase.grid_h * phase.grid_w * phase.n_floors * 0.6)
            ),
            n_agents      = 1,
            obs_radius    = ARCHETYPES[dominant_arch_id].obs_radius,
            comm_mode     = ARCHETYPES[dominant_arch_id].comm_mode,
            msg_dim       = ARCHETYPES[dominant_arch_id].msg_dim,
            reward_mode   = ARCHETYPES[dominant_arch_id].reward_mode,
            reward_alpha  = ARCHETYPES[dominant_arch_id].reward_alpha,
            max_steps     = phase.max_steps,
            reward_wall   = rwall,
            reward_new_cell   = 5.0,
            reward_completion = phase.completion_bonus,
            reward_redundant  = -0.01,
            reward_step       = 0.0,    # sin penalización por paso: deja que reward_new_cell domine
            seed          = 42,
        )
        
        base_env = MARLExploration3D(cfg)
        vec_env = SB3MultiAgentVecEnv(base_env, dominant_arch_id)
        
        ppo_parameters = _ppo_kwargs(phase.max_steps, 1)

        if model is None:
            model = PPO("MlpPolicy", vec_env, **ppo_parameters)
        else:
            model.set_env(vec_env)
            model.n_steps    = ppo_parameters["n_steps"]
            model.batch_size = ppo_parameters["batch_size"]
            model.rollout_buffer = RolloutBuffer(
                model.n_steps,
                model.observation_space,
                model.action_space,
                device    = model.device,
                gamma     = model.gamma,
                gae_lambda= model.gae_lambda,
                n_envs    = model.n_envs,
            )
            # Decaer learning rate al cambiar de fase (preserva lo aprendido)
            model.learning_rate = max(float(model.learning_rate) * 0.8, 5e-5)

        total_timesteps = phase.n_episodes * phase.max_steps
        callback = MARLCurriculumCallback(
            advance_cov=phase.advance_cov, advance_k=phase.advance_k, verbose=1
        )

        model.learn(
            total_timesteps    = total_timesteps,
            callback           = callback,
            reset_num_timesteps= False,
            log_interval       = 20,
        )

        ckpt_name = f"{OUT_DIR}/model_team_{team_id}_phase_{phase.idx}"
        model.save(ckpt_name)
        print(f"✓ Fase {phase.idx} guardada: {ckpt_name}.zip")
        vec_env.close()

if __name__ == "__main__":

    train_team_sb3(team_id=1, start_phase=1, is_fast=False)