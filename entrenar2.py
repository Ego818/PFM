"""
entrenar.py  —  Entrenamiento PPO multi-política con Stable-Baselines3
=======================================================================
Entrena un equipo del MARL de exploración 3D usando PPO de SB3
con curriculum learning progresivo, respetando la composición real
de arquetipos definida en agente_info.py.

ARQUITECTURA MULTI-POLÍTICA (v3 — corregida)
=============================================
Cada arquetipo único del equipo tiene su propio modelo PPO independiente.
Los agentes que comparten arquetipo comparten parámetros (parameter sharing).

Ejemplo — Equipo 8 MIXTO-C [4,4,3,3]:
  · modelo_arch4  →  agentes 0, 1  (CARTÓGRAFO: obs_radius=8, private_map)
  · modelo_arch3  →  agentes 2, 3  (MENSAJERO: msg_dim=8, comm_mode=explicit)

El entorno se construye con los parámetros del arquetipo dominante para
que ExplorationConfig sea coherente en cuanto a canales de observación.
Cada ArchetypeVecEnv aplica AgentObsWrapper a sus agentes para adaptar
la observación cruda al formato correcto de ese arquetipo (mapa privado,
etc.).

CORRECCIONES v3 respecto a la versión anterior
===============================================
 1. train_team_sb3 ahora instancia MultiPolicyTrainer en lugar del
    flujo legacy de política única.
 2. AgentObsWrapper se crea por arquetipo y se aplica en
    ArchetypeVecEnv._extract_obs y en ingest_step, de modo que cada
    modelo recibe observaciones adaptadas a su tipo.
 3. MultiPolicyTrainer.learn() reescrito: loop manual de rollout que
    acumula transiciones en los RolloutBuffers de SB3 usando la API
    interna collect_rollouts-compatible, y llama a train() para
    actualizar pesos cuando el buffer está lleno.
 4. Checkpoints guardados con sufijo _arch{id} (multi-política).
 5. _build_env construye el entorno con parámetros del dominante y
    devuelve también la lista de arquetipos únicos para que el trainer
    cree un VecEnv+PPO por cada uno.

COMPATIBILIDAD CON video.py
============================
Los checkpoints siguen el esquema:
  checkpoints_sb3/model_team{T}_phase{P}_arch{A}.zip
video.py (actualizado) itera sobre todos los .zip por arquetipo.
"""

from __future__ import annotations
import os, sys, argparse
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer

sys.path.insert(0, ".")
from marl_exploration_3d import MARLExploration3D, ExplorationConfig
from agente_info import TEAMS, CURRICULUM, ARCHETYPES, AgentObsWrapper, AgentType

OUT_DIR = "checkpoints_sb3"
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE HIPERPARÁMETROS
# ═══════════════════════════════════════════════════════════════

def _ppo_kwargs(phase_max_steps: int, archetype: AgentType) -> dict:
    """
    Hiperparámetros PPO por arquetipo.

    Se usa el lr propio del arquetipo (definido en AgentType.lr) para
    respetar la diferencia entre arquetipos: CARTÓGRAFO y COLMENA tienen
    lr=4e-4 (red más compleja), el resto 5e-4.

    n_steps = 1024: cubre ~5 episodios de fase-0 por rollout.
    batch_size = 256: buena cobertura con actualizaciones frecuentes.
    gamma = 0.99: horizonte razonable con recompensas normalizadas.
    ent_coef = 0.01: mantiene exploración sin dominar el loss.
    """
    return dict(
        learning_rate       = archetype.lr,
        n_steps             = 1024,
        batch_size          = 256,
        n_epochs            = 10,
        gamma               = 0.99,
        gae_lambda          = 0.95,
        clip_range          = 0.2,
        ent_coef            = 0.01,
        vf_coef             = 0.5,
        max_grad_norm       = 0.5,
        normalize_advantage = True,
        policy_kwargs       = dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose             = 1,
        device              = "auto",
    )


# ═══════════════════════════════════════════════════════════════
# 1. WRAPPER MARL → VecEnv (SB3) — por arquetipo
# ═══════════════════════════════════════════════════════════════

class ArchetypeVecEnv(VecEnv):
    """
    VecEnv de SB3 para un subconjunto de agentes que comparten arquetipo.

    Expone SOLO los agentes de `agent_indices` como sub-entornos virtuales.
    El entorno base (MARLExploration3D) es compartido con los demás
    ArchetypeVecEnv del mismo equipo; el paso real lo gestiona
    MultiPolicyTrainer, no este wrapper.

    OBSERVACIONES ADAPTADAS POR ARQUETIPO
    ──────────────────────────────────────
    Cada ArchetypeVecEnv mantiene un AgentObsWrapper para su arquetipo.
    - En _extract_obs se aplica AgentObsWrapper.process() a cada agente,
      lo que inyecta el mapa privado (si private_map=True) en el vector
      de observación antes de pasarlo al modelo PPO.
    - En ingest_step se llama a update_private_map() para que el wrapper
      registre la celda que cada agente acaba de visitar.
    - En reset_wrappers() se reinician los mapas privados al inicio de
      cada episodio.

    NORMALIZACIÓN DE RECOMPENSAS
    ─────────────────────────────
    Welford online independiente por arquetipo.
    """

    def __init__(
        self,
        env:           MARLExploration3D,
        archetype_id:  int,
        agent_indices: List[int],
        obs_dict_init: Dict[int, np.ndarray],
    ):
        self.env           = env
        self.arch          = ARCHETYPES[archetype_id]
        self.archetype_id  = archetype_id
        self.agent_indices = agent_indices
        self.info: Dict    = {}

        # ── AgentObsWrapper para este arquetipo ──────────────────
        # n_agents = total de agentes del entorno (para dimensionar
        # los mapas privados por índice global).
        self.obs_wrapper = AgentObsWrapper(
            archetype = self.arch,
            n_agents  = env.n_agents,
        )
        self.obs_wrapper.reset(env, agent_indices)

        # Dimensión real de obs tras aplicar el wrapper
        sample_raw = obs_dict_init[agent_indices[0]]
        sample_processed = self.obs_wrapper.process(sample_raw, agent_indices[0], env)
        obs_dim = sample_processed.shape

        observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_dim, dtype=np.float32
        )
        action_space = gym.spaces.Discrete(env.n_actions)

        self.metadata    = {"render_modes": ["rgb_array"]}
        self.render_mode = "rgb_array"

        super().__init__(
            num_envs          = len(agent_indices),
            observation_space = observation_space,
            action_space      = action_space,
        )

        # Buffer interno: el trainer deposita aquí los datos del paso
        self._pending_obs:    Optional[np.ndarray] = self._extract_obs(obs_dict_init)
        self._pending_rews:   Optional[np.ndarray] = None
        self._pending_dones:  Optional[np.ndarray] = None
        self._pending_infos:  Optional[List[Dict]] = None
        self._actions:        Dict[int, int]        = {}

        # Normalización Welford independiente por arquetipo
        self._rew_mean  = 0.0
        self._rew_var   = 1.0
        self._rew_count = 1e-4

    # ── Normalización ────────────────────────────────────────────

    def _normalize_rewards(self, rews: np.ndarray) -> np.ndarray:
        for r in rews:
            self._rew_count += 1
            delta = r - self._rew_mean
            self._rew_mean += delta / self._rew_count
            self._rew_var  += (delta * (r - self._rew_mean) - self._rew_var) / self._rew_count
        std = float(np.sqrt(max(self._rew_var, 1e-8)))
        return np.clip((rews - self._rew_mean) / std, -10.0, 10.0).astype(np.float32)

    # ── Extracción de obs con wrapper de arquetipo ───────────────

    def _extract_obs(self, obs_dict: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Aplica AgentObsWrapper.process() a cada agente de este arquetipo
        antes de apilar las observaciones. Esto garantiza que el modelo
        recibe la vista correcta (con/sin mapa privado) para su tipo.
        """
        processed = [
            self.obs_wrapper.process(obs_dict[i], i, self.env)
            for i in self.agent_indices
        ]
        return np.array(processed, dtype=np.float32)

    # ── Reinicio de mapas privados al inicio de episodio ─────────

    def reset_wrappers(self) -> None:
        """Llama al reset del AgentObsWrapper para limpiar mapas privados."""
        self.obs_wrapper.reset(self.env, self.agent_indices)

    # ── API llamada por MultiPolicyTrainer ───────────────────────

    def get_obs(self) -> np.ndarray:
        """Devuelve la observación actual (ya procesada) del subconjunto."""
        assert self._pending_obs is not None
        return self._pending_obs

    def collect_actions(self, model: "PPO") -> Dict[int, int]:
        """Infiere acciones para los agentes de este arquetipo."""
        obs = self.get_obs()
        actions_arr, _ = model.predict(obs, deterministic=False)
        return {
            global_i: int(a)
            for global_i, a in zip(self.agent_indices, actions_arr)
        }

    def ingest_step(
        self,
        obs_next:  Dict[int, np.ndarray],
        rewards:   Dict[int, float],
        done:      bool,
        info:      Dict,
        reset_obs: Optional[Dict[int, np.ndarray]] = None,
    ) -> None:
        """
        Deposita en el buffer interno los resultados del paso del entorno.
        Llamado por MultiPolicyTrainer tras env.step().

        Si done=True:
          - Actualiza el mapa privado con la posición PRE-reset.
          - Reinicia los wrappers (borra mapas privados).
          - Usa reset_obs (obs post-reset) como obs para el siguiente step.
        Si done=False:
          - Actualiza el mapa privado con la posición actual.
        """
        self.info = info

        # Actualizar mapa privado ANTES del posible reset
        for i in self.agent_indices:
            self.obs_wrapper.update_private_map(i, self.env)

        if done:
            # Añadir penalización de redundancia del arquetipo (si aplica)
            rews_raw = np.array(
                [rewards[i] + self.obs_wrapper.redundancy_penalty(i, self.env)
                 for i in self.agent_indices],
                dtype=np.float32,
            )
            # Reiniciar wrappers para el nuevo episodio
            self.reset_wrappers()
            # obs post-reset ya procesada
            obs_src = reset_obs if reset_obs is not None else obs_next
            obs_arr = self._extract_obs(obs_src)
        else:
            rews_raw = np.array(
                [rewards[i] + self.obs_wrapper.redundancy_penalty(i, self.env)
                 for i in self.agent_indices],
                dtype=np.float32,
            )
            obs_arr = self._extract_obs(obs_next)

        rews_arr  = self._normalize_rewards(rews_raw)
        dones_arr = np.array([done] * len(self.agent_indices), dtype=bool)
        infos     = [info.copy() for _ in self.agent_indices]

        if done:
            # SB3 necesita terminal_observation con la obs del último step
            terminal_obs = self._extract_obs(obs_next)
            for idx in range(len(self.agent_indices)):
                infos[idx]["terminal_observation"] = terminal_obs[idx].copy()
                infos[idx]["TimeLimit.truncated"]  = bool(
                    info.get("TimeLimit.truncated", False)
                )

        self._pending_obs   = obs_arr
        self._pending_rews  = rews_arr
        self._pending_dones = dones_arr
        self._pending_infos = infos

    # ── API VecEnv requerida por SB3 ────────────────────────────

    def reset(self, **kwargs) -> np.ndarray:
        assert self._pending_obs is not None
        return self._pending_obs

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = {
            global_i: int(a)
            for global_i, a in zip(self.agent_indices, actions)
        }

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        assert self._pending_rews is not None, "ingest_step no llamado antes de step_wait"
        obs   = self._pending_obs
        rews  = self._pending_rews
        dones = self._pending_dones
        infos = self._pending_infos
        self._pending_rews  = None
        self._pending_dones = None
        self._pending_infos = None
        return obs, rews, dones, infos

    # ── Métodos abstractos restantes ─────────────────────────────

    def close(self) -> None:
        pass

    def get_attr(self, attr_name: str, indices=None) -> List[Any]:
        val = getattr(self, attr_name, None)
        n   = self.num_envs if indices is None else len(indices)
        return [val] * n

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        setattr(self, attr_name, value)

    def env_method(self, method_name: str, *args, indices=None, **kwargs) -> List[Any]:
        n = self.num_envs if indices is None else len(indices)
        return [None] * n

    def env_is_wrapped(self, wrapper_class: type, indices=None) -> List[bool]:
        n = self.num_envs if indices is None else len(indices)
        return [False] * n


# Alias de compatibilidad con video.py (modo legado)
SB3MultiAgentVecEnv = ArchetypeVecEnv


# ═══════════════════════════════════════════════════════════════
# 1b. ORQUESTADOR MULTI-POLÍTICA
# ═══════════════════════════════════════════════════════════════

class MultiPolicyTrainer:
    """
    Orquesta N modelos PPO (uno por arquetipo único del equipo).

    En cada paso de rollout:
      1. Pide acciones a cada modelo para sus agentes.
      2. Combina todas las acciones y ejecuta un único env.step().
      3. Distribuye obs/rewards/done a cada ArchetypeVecEnv.
      4. Cuando el RolloutBuffer de un modelo está lleno, SB3
         actualiza sus pesos (llamada a model.train()).

    El loop de rollout está implementado manualmente (no delegamos a
    model.learn()) para poder sincronizar el paso del entorno entre
    todos los arquetipos y para inyectar los datos correctamente en
    cada VecEnv a través de ingest_step.
    """

    def __init__(
        self,
        base_env:   MARLExploration3D,
        team_id:    int,
        ppo_params_per_arch: Dict[int, dict],
    ):
        self.env  = base_env
        self.team = TEAMS[team_id]

        # Mapear arquetipo → lista de índices globales
        self.arch_to_agents: Dict[int, List[int]] = {}
        for agent_idx, arch_id in enumerate(self.team.composition):
            self.arch_to_agents.setdefault(arch_id, []).append(agent_idx)

        # Reset inicial
        obs_dict, _ = self.env.reset()

        # Crear ArchetypeVecEnv + PPO por arquetipo
        self.vec_envs: Dict[int, ArchetypeVecEnv] = {}
        self.models:   Dict[int, PPO]              = {}

        for arch_id, agent_indices in self.arch_to_agents.items():
            vec_env = ArchetypeVecEnv(
                env           = self.env,
                archetype_id  = arch_id,
                agent_indices = agent_indices,
                obs_dict_init = obs_dict,
            )
            params = ppo_params_per_arch[arch_id]
            model  = PPO("MlpPolicy", vec_env, **params)
            self.vec_envs[arch_id] = vec_env
            self.models[arch_id]   = model

        self._current_obs: Dict[int, np.ndarray] = obs_dict

    # ── Cambio de fase (nuevo entorno, mismos pesos) ─────────────

    def set_phase_env(
        self,
        base_env: MARLExploration3D,
        ppo_params_per_arch: Dict[int, dict],
    ) -> None:
        """
        Sustituye el entorno base al cambiar de fase.
        Transfiere los pesos de los modelos y reconstruye los buffers.
        """
        self.env = base_env
        obs_dict, _ = self.env.reset()
        self._current_obs = obs_dict

        for arch_id, agent_indices in self.arch_to_agents.items():
            new_vec = ArchetypeVecEnv(
                env           = self.env,
                archetype_id  = arch_id,
                agent_indices = agent_indices,
                obs_dict_init = obs_dict,
            )
            self.vec_envs[arch_id] = new_vec
            model  = self.models[arch_id]
            params = ppo_params_per_arch[arch_id]
            model.set_env(new_vec)
            model.n_steps    = params["n_steps"]
            model.batch_size = params["batch_size"]
            _rebuild_rollout_buffer(model)
            new_lr = _apply_lr_decay(model, factor=0.8, min_lr=5e-5)
            print(
                f"    Arquetipo {arch_id} ({ARCHETYPES[arch_id].name}): "
                f"lr → {new_lr:.2e}"
            )

    # ── Paso sincronizado ────────────────────────────────────────

    def step_all(self) -> Tuple[bool, Dict]:
        """
        Recolecta acciones de todos los modelos, ejecuta un paso del
        entorno y distribuye los resultados. Devuelve (done, info).
        """
        # 1. Recolectar acciones de cada arquetipo
        all_actions: Dict[int, int] = {}
        for arch_id, model in self.models.items():
            vec_env = self.vec_envs[arch_id]
            actions = vec_env.collect_actions(model)
            all_actions.update(actions)

        # 2. Paso del entorno
        obs_next, rewards, terminated, truncated, info = self.env.step(all_actions)
        done = terminated or truncated

        # 3. Si terminó, hacer reset y pasar obs post-reset a ingest_step
        reset_obs = None
        if done:
            reset_obs, _ = self.env.reset()
            self._current_obs = reset_obs
        else:
            self._current_obs = obs_next

        # 4. Distribuir resultados a cada ArchetypeVecEnv
        for arch_id, vec_env in self.vec_envs.items():
            vec_env.ingest_step(obs_next, rewards, done, info, reset_obs)

        return done, info

    # ── Entrenamiento sincronizado ───────────────────────────────

    def learn(
        self,
        total_timesteps: int,
        callback: "MARLCurriculumCallback",
    ) -> None:
        """
        Loop manual de rollout multi-política.

        Por cada step del entorno:
          - Todos los modelos dan acciones (step_all).
          - Los datos se inyectan en los ArchetypeVecEnv vía ingest_step.
          - Cada modelo consume su transition a través de step_async/step_wait
            y acumula en su RolloutBuffer interno de SB3.
          - Cuando el buffer de un modelo se llena (n_steps transiciones),
            SB3 llama a train() internamente si usamos collect_rollouts.
            Aquí lo simulamos: cuando acumulamos n_steps, forzamos el update
            llamando a model.train() directamente.

        Este diseño evita el doble-entrenamiento del bug anterior (donde se
        llamaba a model.learn() con total_timesteps al final, volviendo a
        ejecutar el entorno completo).
        """
        # Inicializar contadores de steps por modelo
        steps_in_buffer: Dict[int, int] = {arch_id: 0 for arch_id in self.models}
        steps_done = 0

        # Preparar los modelos para el loop manual
        for arch_id, model in self.models.items():
            model.policy.set_training_mode(True)
            model._last_obs        = self.vec_envs[arch_id].get_obs()
            model._last_episode_starts = np.ones(
                (self.vec_envs[arch_id].num_envs,), dtype=bool
            )
            model.rollout_buffer.reset()

        while steps_done < total_timesteps:
            done, info = self.step_all()
            steps_done += 1

            # Alimentar el buffer de SB3 de cada modelo con la transición
            for arch_id, model in self.models.items():
                vec_env = self.vec_envs[arch_id]

                # step_async/step_wait: SB3 consume la transición pre-inyectada
                vec_env.step_async(
                    np.array(
                        [vec_env._actions.get(i, 0) for i in vec_env.agent_indices],
                        dtype=np.int64,
                    )
                )
                new_obs, rewards_buf, dones_buf, infos_buf = vec_env.step_wait()

                # Añadir al RolloutBuffer de SB3
                # (equivalente a lo que hace collect_rollouts internamente)
                model.rollout_buffer.add(
                    model._last_obs,
                    np.array(
                        [vec_env._actions.get(i, 0) for i in vec_env.agent_indices],
                        dtype=np.int64,
                    ).reshape(-1, 1),
                    rewards_buf,
                    model._last_episode_starts,
                    model.policy.evaluate_actions(
                        model.policy.obs_to_tensor(model._last_obs)[0],
                        model.policy.obs_to_tensor(
                            np.array(
                                [vec_env._actions.get(i, 0)
                                 for i in vec_env.agent_indices],
                                dtype=np.int64,
                            )
                        )[0],
                    )[1],  # values
                    model.policy.evaluate_actions(
                        model.policy.obs_to_tensor(model._last_obs)[0],
                        model.policy.obs_to_tensor(
                            np.array(
                                [vec_env._actions.get(i, 0)
                                 for i in vec_env.agent_indices],
                                dtype=np.int64,
                            )
                        )[0],
                    )[2],  # log_probs
                )

                model._last_obs             = new_obs
                model._last_episode_starts  = dones_buf
                steps_in_buffer[arch_id]   += 1

                # Cuando el buffer está lleno → actualizar pesos
                if steps_in_buffer[arch_id] >= model.n_steps:
                    with th.no_grad():
                        last_values = model.policy.predict_values(
                            model.policy.obs_to_tensor(new_obs)[0]
                        )
                    model.rollout_buffer.compute_returns_and_advantage(
                        last_values = last_values,
                        dones       = dones_buf,
                    )
                    model.train()
                    model.rollout_buffer.reset()
                    steps_in_buffer[arch_id] = 0

            # Criterio de avance de fase (callback)
            if done and callback is not None:
                cov = float(info.get("coverage_ratio", 0.0))
                callback.recent_covs.append(cov)
                if len(callback.recent_covs) > callback.advance_k:
                    callback.recent_covs.pop(0)
                mean_cov = float(np.mean(callback.recent_covs))
                print(
                    f"  [FIN EP] cov={cov*100:.1f}%  "
                    f"media({callback.advance_k})={mean_cov*100:.1f}%"
                )
                if (len(callback.recent_covs) >= callback.advance_k
                        and mean_cov >= callback.advance_cov):
                    print("  ⚡ Criterio de avance alcanzado — siguiente fase.")
                    break

    def save(self, base_path: str) -> None:
        """Guarda un checkpoint por arquetipo."""
        for arch_id, model in self.models.items():
            path = f"{base_path}_arch{arch_id}"
            model.save(path)
            print(
                f"    ✓ Arquetipo {arch_id} "
                f"({ARCHETYPES[arch_id].name}): {path}.zip"
            )

    def close(self) -> None:
        self.env.close()


# ═══════════════════════════════════════════════════════════════
# 2. CALLBACK DE SEGUIMIENTO Y CURRICULUM
# ═══════════════════════════════════════════════════════════════

class MARLCurriculumCallback(BaseCallback):
    """
    Callback ligero usado solo para el criterio de avance de fase.
    El loop de rollout de MultiPolicyTrainer lo consulta directamente.
    """

    def __init__(self, advance_cov: float, advance_k: int, verbose: int = 0):
        super().__init__(verbose)
        self.advance_cov  = advance_cov
        self.advance_k    = advance_k
        self.recent_covs: List[float] = []

    def _on_step(self) -> bool:
        return True


# ═══════════════════════════════════════════════════════════════
# 3. UTILIDADES DE ENTORNO Y MODELOS
# ═══════════════════════════════════════════════════════════════

def _build_env(phase, team_id: int) -> Tuple[MARLExploration3D, int]:
    """
    Construye el entorno base para una fase y equipo dados.

    El entorno se configura con los parámetros del arquetipo DOMINANTE
    del equipo (mayor prioridad según TeamConfig.dominant). Esto define
    el espacio de observación global (obs_radius, comm_mode, msg_dim).

    Cada ArchetypeVecEnv aplica después AgentObsWrapper para adaptar
    la observación a las capacidades reales de cada arquetipo (p.ej.
    inyectar el mapa privado del CARTÓGRAFO).

    Devuelve (base_env, dominant_arch_id).
    """
    team          = TEAMS[team_id]
    dominant_arch = team.dominant
    n_agents      = team.n_agents

    rwall = -0.1 if phase.idx <= 1 else -0.3

    cfg = ExplorationConfig(
        grid_h        = phase.grid_h,
        grid_w        = phase.grid_w,
        n_floors      = phase.n_floors,
        wall_density  = phase.wall_density,
        n_stairs      = phase.n_stairs,
        min_connected = min(
            phase.min_connected,
            int(phase.grid_h * phase.grid_w * phase.n_floors * 0.6),
        ),
        n_agents      = n_agents,
        obs_radius    = dominant_arch.obs_radius,
        comm_mode     = dominant_arch.comm_mode,
        msg_dim       = dominant_arch.msg_dim,
        reward_mode   = dominant_arch.reward_mode,
        reward_alpha  = dominant_arch.reward_alpha,
        max_steps     = phase.max_steps,
        reward_wall       = rwall,
        reward_new_cell   = 1.0,
        reward_completion = phase.completion_bonus / 70.0,
        reward_redundant  = -0.01,
        reward_step       = -0.005,
        seed          = 42,
    )

    base_env = MARLExploration3D(cfg)
    return base_env, dominant_arch.id


def _rebuild_rollout_buffer(model: PPO) -> None:
    """Recrea el RolloutBuffer con los n_steps actuales del modelo."""
    model.rollout_buffer = RolloutBuffer(
        buffer_size       = model.n_steps,
        observation_space = model.observation_space,
        action_space      = model.action_space,
        device            = model.device,
        gamma             = model.gamma,
        gae_lambda        = model.gae_lambda,
        n_envs            = model.n_envs,
    )


def _apply_lr_decay(model: PPO, factor: float = 0.8, min_lr: float = 5e-5) -> float:
    """Aplica decay al lr y actualiza el optimizador de PyTorch."""
    old_lr = (
        float(model.learning_rate)
        if not callable(model.learning_rate)
        else model.learning_rate(1.0)
    )
    new_lr = max(old_lr * factor, min_lr)
    model.learning_rate = new_lr
    if hasattr(model, "policy") and model.policy is not None:
        for pg in model.policy.optimizer.param_groups:
            pg["lr"] = new_lr
    return new_lr


# ═══════════════════════════════════════════════════════════════
# 4. PROCESO PRINCIPAL DE ENTRENAMIENTO
# ═══════════════════════════════════════════════════════════════

def train_team_sb3(team_id: int, start_phase: int = 0):
    """
    Entrena el equipo `team_id` a través del curriculum completo.

    Usa MultiPolicyTrainer para gestionar un modelo PPO independiente
    por cada arquetipo único del equipo. Las observaciones de cada agente
    pasan por AgentObsWrapper antes de llegar al modelo correspondiente.

    Parámetros
    ----------
    team_id     : ID del equipo (6–10 con la configuración actual de TEAMS).
    start_phase : Índice de fase desde la que arrancar (0 = desde el principio).
    """
    team = TEAMS[team_id]

    print(f"\n{'='*70}")
    print(f"  EQUIPO {team_id}: {team.name}  {team.label}")
    print(f"  {team.description}")
    print(f"  Agentes: {team.n_agents}  |  Dominante: {team.dominant.name}")
    unique_ids = team.unique_archetypes()
    print(f"  Arquetipos únicos: {[ARCHETYPES[a].name for a in unique_ids]}")
    print(f"{'='*70}\n")

    trainer: Optional[MultiPolicyTrainer] = None

    for phase in CURRICULUM:

        if phase.idx < start_phase:
            print(f"  [SKIP] Fase {phase.idx}: {phase.name}  (< start_phase={start_phase})")
            continue

        if phase.n_episodes == 0:
            print(f"  [SKIP] Fase {phase.idx}: {phase.name}  (n_episodes=0)")
            continue

        print(f"\n{'='*70}")
        print(
            f"🚀 FASE {phase.idx}: {phase.name}  |  "
            f"Mapa: {phase.grid_h}×{phase.grid_w}×{phase.n_floors}  |  "
            f"Episodios: {phase.n_episodes}  |  "
            f"Max pasos/ep: {phase.max_steps}"
        )
        print(f"{'='*70}")

        # ── Construir entorno ─────────────────────────────────────
        base_env, dominant_arch_id = _build_env(phase, team_id)

        # Hiperparámetros PPO por arquetipo (lr propio de cada AgentType)
        ppo_params_per_arch = {
            arch_id: _ppo_kwargs(phase.max_steps, ARCHETYPES[arch_id])
            for arch_id in team.unique_archetypes()
        }

        # ── Inicializar o actualizar el trainer ───────────────────
        if trainer is None:
            trainer = MultiPolicyTrainer(
                base_env             = base_env,
                team_id              = team_id,
                ppo_params_per_arch  = ppo_params_per_arch,
            )
            print(f"  MultiPolicyTrainer creado.")
            for arch_id in unique_ids:
                p = ppo_params_per_arch[arch_id]
                print(
                    f"    · {ARCHETYPES[arch_id].name}  "
                    f"agentes={trainer.arch_to_agents[arch_id]}  "
                    f"lr={p['learning_rate']:.1e}"
                )
        else:
            print(f"  Actualizando entorno para fase {phase.idx}...")
            trainer.set_phase_env(base_env, ppo_params_per_arch)

        # ── Entrenamiento ─────────────────────────────────────────
        total_timesteps = phase.n_episodes * phase.max_steps
        callback = MARLCurriculumCallback(
            advance_cov = phase.advance_cov,
            advance_k   = phase.advance_k,
            verbose     = 1,
        )

        print(
            f"  Entrenando {total_timesteps:,} timesteps "
            f"({phase.n_episodes} eps × {phase.max_steps} pasos) ..."
        )
        trainer.learn(
            total_timesteps = total_timesteps,
            callback        = callback,
        )

        # ── Checkpoint ───────────────────────────────────────────
        ckpt_base = f"{OUT_DIR}/model_team{team_id}_phase{phase.idx}"
        print(f"  Guardando checkpoints: {ckpt_base}_arch*.zip")
        trainer.save(ckpt_base)

        base_env.close()

    print(f"\n{'='*70}")
    print(f"  Entrenamiento completo — Equipo {team_id}: {team.name}")
    print(f"  Checkpoints en: {OUT_DIR}/")
    print(f"{'='*70}\n")


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrenamiento PPO multi-política (SB3) — MARL Exploración 3D"
    )
    parser.add_argument(
        "--team", type=int, default=6,
        help="ID del equipo a entrenar (6-10, default=6)"
    )
    parser.add_argument(
        "--fase", type=int, default=0,
        help="Fase de inicio del curriculum (0-5, default=0)"
    )
    args = parser.parse_args()

    train_team_sb3(team_id=args.team, start_phase=args.fase)