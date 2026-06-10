"""
entrenar.py  —  Entrenamiento PPO con Stable-Baselines3
========================================================
Entrena un equipo del MARL de exploración 3D usando PPO de SB3
con curriculum learning progresivo.

CORRECCIONES v2 (por qué no aprendía antes)
============================================
 1. NORMALIZACIÓN DE RECOMPENSAS en el wrapper (Welford online).
    reward_new_cell=70 producía recompensas crudas de miles por episodio.
    SB3 no normaliza por defecto → value loss dominaba el entrenamiento
    → gradiente de política nulo → cobertura plana en ~1.6% para siempre.

 2. reward_new_cell = 1.0 (no 70.0).
    Con el normalizador online, la escala cruda no importa tanto, pero
    1.0 hace que la señal sea comparable a las penalizaciones desde el
    inicio, evitando oscilaciones en el running_std.

 3. reward_step = -0.005 (no 0.0).
    Sin coste de paso la política aleatoria es indiferente a moverse.
    Un coste pequeño crea presión para encontrar celdas nuevas.

 4. reward_wall = -0.1 fases 0-1 (no 0.0).
    Sin penalización de pared los agentes pasan el 30-40% de pasos
    chocando, reduciendo la señal de exploración efectiva.

 5. gamma = 0.99 (no 0.995).
    Con recompensas normalizadas ~1.0, gamma=0.995 infla los retornos
    descontados (~200 con 200 pasos). gamma=0.99 es adecuado.

 6. ent_coef = 0.01 (no 0.05).
    Con recompensas normalizadas, 0.05 aplastaba el gradiente de política.

 7. n_steps = 1024 (no 2048).
    Cubre ~5 episodios de fase-0 (max_steps=200) por rollout, con menos
    varianza entre episodios en el mismo batch.

 8. Callback: cobertura leída de locals["infos"][0] (no de get_attr("info")).
    get_attr con nuestro VecEnv personalizado devolvía el atributo del
    wrapper que podía estar desactualizado tras el auto-reset.

 9. n_agents correcto: se toma del TeamConfig.
10. Decaimiento de lr al cambiar de fase con actualización del optimizador PyTorch.
11. Reconstrucción del RolloutBuffer con n_steps actualizado al cambiar de fase.
"""

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
from agente_info import TEAMS, CURRICULUM, ARCHETYPES, AgentObsWrapper

OUT_DIR = "checkpoints_sb3"
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE HIPERPARÁMETROS
# ═══════════════════════════════════════════════════════════════

def _ppo_kwargs(max_steps: int, n_agents: int) -> dict:
    """
    Hiperparámetros PPO calibrados para recompensas normalizadas (escala ~1.0).

    Cambios clave respecto a versión anterior:
      - n_steps = 1024: cubre ~5 episodios de fase-0 (max_steps=200) por rollout.
        2048 era excesivo: demasiada varianza entre episodios en el mismo rollout.
      - batch_size = 256: buena cobertura del rollout con actualizaciones frecuentes.
      - gamma = 0.99: horizonte razonable. 0.995 infla retornos con recompensas
        normalizadas → ventajas enormes → gradientes inestables.
      - ent_coef = 0.01: con recompensas normalizadas, 0.05 aplastaba el gradiente
        de política. 0.01 mantiene exploración sin dominar el loss.
      - vf_coef = 0.5: estándar PPO. El value loss ya está controlado por la
        normalización de recompensas en el wrapper.
      - normalize_advantage = True (activo por defecto en SB3): ventajas
        normalizadas por batch → gradientes estables independientemente de la escala.
    """
    n_steps  = 1024
    batch_sz = 256

    return dict(
        learning_rate  = 3e-4,
        n_steps        = n_steps,
        batch_size     = batch_sz,
        n_epochs       = 10,
        gamma          = 0.99,
        gae_lambda     = 0.95,
        clip_range     = 0.2,
        ent_coef       = 0.01,
        vf_coef        = 0.5,
        max_grad_norm  = 0.5,
        normalize_advantage = True,
        policy_kwargs  = dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose        = 1,
        device         = "auto",
    )


# ═══════════════════════════════════════════════════════════════
# 1. WRAPPER MARL → VecEnv (SB3)
# ═══════════════════════════════════════════════════════════════

class SB3MultiAgentVecEnv(VecEnv):
    """
    Adapta MARLExploration3D (n_agents) a la interfaz VecEnv de SB3
    exponiendo cada agente como un "sub-entorno" virtual.

    Todas las acciones se envían juntas al entorno real en step_wait,
    y las observaciones/recompensas de cada agente se devuelven como
    filas independientes del batch (num_envs = n_agents).

    Cuando el episodio termina, el auto-reset se realiza internamente
    y la observación devuelta es la del nuevo episodio (estándar SB3).

    NORMALIZACIÓN DE RECOMPENSAS
    ─────────────────────────────
    Las recompensas crudas del entorno (reward_new_cell=70) tienen escala
    muy grande para PPO. Se normalizan online con running mean/std usando
    el algoritmo de Welford, igual que hace VecNormalize de SB3 pero sin
    necesidad de envolver el env en dos capas.
    El valor normalizado se clipea a [-10, 10] para estabilidad.
    La normalización NO se aplica a la info (coverage_ratio) — solo a las
    recompensas que ve PPO.
    """

    def __init__(self, env: MARLExploration3D, archetype_id: int):
        self.env           = env
        self.arch          = ARCHETYPES[archetype_id]
        self.n_agents      = env.cfg.n_agents
        self.agent_indices = list(range(self.n_agents))
        self.info: Dict    = {}

        # Medir dimensión real de observación con un reset de prueba
        obs_p, _ = self.env.reset()
        obs_dim  = obs_p[0].shape

        observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_dim, dtype=np.float32
        )
        action_space = gym.spaces.Discrete(self.env.n_actions)

        self.metadata    = {"render_modes": ["rgb_array"]}
        self.render_mode = "rgb_array"

        super().__init__(
            num_envs          = self.n_agents,
            observation_space = observation_space,
            action_space      = action_space,
        )
        self.actions_dict: Dict[int, int] = {}

        # ── Normalización online de recompensas (Welford) ────────
        self._rew_mean  = 0.0
        self._rew_var   = 1.0
        self._rew_count = 1e-4   # evita división por cero al inicio

    def _normalize_rewards(self, rews: np.ndarray) -> np.ndarray:
        """Normaliza recompensas con running mean/std. Clip a [-10, 10]."""
        for r in rews:
            self._rew_count += 1
            delta = r - self._rew_mean
            self._rew_mean += delta / self._rew_count
            self._rew_var  += (delta * (r - self._rew_mean) - self._rew_var) / self._rew_count
        std = float(np.sqrt(max(self._rew_var, 1e-8)))
        return np.clip((rews - self._rew_mean) / std, -10.0, 10.0).astype(np.float32)

    # ── API pública ──────────────────────────────────────────────

    def reset(self, **kwargs) -> np.ndarray:
        seed = kwargs.get("seed", None)
        if seed is not None:
            obs_dict, self.info = self.env.reset(seed=seed)
        else:
            obs_dict, self.info = self.env.reset()
        return self._get_obs(obs_dict)

    def step_async(self, actions: np.ndarray) -> None:
        self.actions_dict = {i: int(act) for i, act in enumerate(actions)}

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Ejecuta un paso del entorno con las acciones almacenadas.
        Al finalizar el episodio realiza auto-reset (estándar VecEnv de SB3).
        """
        obs_next, rewards, terminated, truncated, info = self.env.step(
            self.actions_dict
        )

        self.info = info
        done      = terminated or truncated
        dones     = np.array([done] * self.n_agents, dtype=bool)
        rews_raw  = np.array([rewards[i] for i in self.agent_indices], dtype=np.float32)
        rews_arr  = self._normalize_rewards(rews_raw)
        obs_arr   = self._get_obs(obs_next)
        infos     = [info.copy() for _ in range(self.n_agents)]

        if done:
            # Guardar la última obs del episodio antes del reset (para SB3)
            for idx in range(self.n_agents):
                infos[idx]["terminal_observation"] = obs_arr[idx].copy()
                infos[idx]["TimeLimit.truncated"]  = bool(truncated)
            obs_arr = self.reset()

        return obs_arr, rews_arr, dones, infos

    # ── Métodos abstractos de VecEnv ────────────────────────────

    def close(self) -> None:
        self.env.close()

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

    # ── Interno ──────────────────────────────────────────────────

    def _get_obs(self, obs_dict) -> np.ndarray:
        return np.array(
            [obs_dict[i] for i in self.agent_indices], dtype=np.float32
        )


# ═══════════════════════════════════════════════════════════════
# 2. CALLBACK DE SEGUIMIENTO Y CURRICULUM
# ═══════════════════════════════════════════════════════════════

class MARLCurriculumCallback(BaseCallback):
    """
    Callback que:
     - Acumula la recompensa total del equipo por episodio y la imprime.
     - Mantiene una ventana deslizante de `advance_k` coberturas para
       detectar cuándo la política es suficientemente buena para
       avanzar de fase (retorna False → detiene model.learn).

    CORRECCIÓN: la cobertura se lee de locals["infos"][0] (el dict de info
    del paso actual), no de training_env.get_attr("info") que con nuestro
    VecEnv personalizado devuelve el atributo del wrapper y puede estar
    desactualizado cuando hay auto-reset.
    """

    def __init__(self, advance_cov: float, advance_k: int, verbose: int = 0):
        super().__init__(verbose)
        self.advance_cov  = advance_cov
        self.advance_k    = advance_k
        self.recent_covs: List[float] = []
        self._ep_rew_acc  = 0.0   # acumulador para el episodio en curso

    def _on_step(self) -> bool:
        # Acumular recompensas de TODOS los agentes en este paso
        rewards = self.locals.get("rewards", None)
        if rewards is not None:
            self._ep_rew_acc += float(np.sum(rewards))

        # Detectar fin de episodio (dones es un array bool por agente)
        dones = self.locals.get("dones", np.array([False]))
        if dones.any():
            # Leer info directamente del paso actual.
            # locals["infos"] es una lista con un dict por sub-env.
            # Cuando dones[0]=True, infos[0] contiene los datos del episodio
            # que ACABA DE TERMINAR (antes del auto-reset en step_wait).
            infos = self.locals.get("infos", [{}])
            info  = infos[0] if infos else {}

            cov = float(info.get("coverage_ratio", 0.0))
            self.recent_covs.append(cov)

            # Ventana deslizante de tamaño advance_k
            if len(self.recent_covs) > self.advance_k:
                self.recent_covs.pop(0)

            mean_cov = float(np.mean(self.recent_covs))

            print(
                f"🟩 [FIN EPISODIO] "
                f"Cobertura: {cov*100:.2f}%  |  "
                f"Media ({self.advance_k}ep): {mean_cov*100:.2f}%  |  "
                f"Rew. total equipo: {self._ep_rew_acc:.2f}"
            )

            # Resetear acumulador para el próximo episodio
            self._ep_rew_acc = 0.0

            # ¿Avanzar de fase?
            if (len(self.recent_covs) >= self.advance_k
                    and mean_cov >= self.advance_cov):
                print(
                    f"│ ⚡ Criterio de avance alcanzado "
                    f"(media {mean_cov*100:.2f}% ≥ objetivo {self.advance_cov*100:.2f}%) "
                    f"— pasando a la siguiente fase."
                )
                return False   # detiene model.learn

        return True


# ═══════════════════════════════════════════════════════════════
# 3. PROCESO PRINCIPAL DE ENTRENAMIENTO
# ═══════════════════════════════════════════════════════════════

def _build_env(phase, team_id: int) -> Tuple[MARLExploration3D, int]:
    """
    Construye el entorno para una fase y equipo dados.

    ESCALA DE RECOMPENSAS
    ─────────────────────
    reward_new_cell = 1.0  (no 70.0): el wrapper normaliza online, así que
    la escala cruda solo afecta al running_std. Usar 1.0 hace que la señal
    sea comparable a la penalización por paso y al castigo por pared, lo que
    da gradientes equilibrados desde el primer rollout.

    reward_step = -0.005: penalización pequeña por paso. Sin coste de paso
    la política sin aprendizaje (moverse aleatorio o quedarse quieto) es
    indiferente; con este coste hay presión para encontrar celdas nuevas.

    reward_wall = -0.1 en fases tempranas (no 0.0): sin penalización por
    pared los agentes pasan el 30-40% de pasos chocando, lo que reduce la
    señal de exploración efectiva.
    """
    team           = TEAMS[team_id]
    dominant_arch  = team.dominant
    n_agents       = team.n_agents
    dominant_id    = dominant_arch.id

    # reward_wall: leve pero no nulo — evita comportamiento de rebotar en paredes
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
        reward_new_cell   = 1.0,      # escala base; el wrapper normaliza online
        reward_completion = phase.completion_bonus / 70.0,  # reescalar igual que new_cell
        reward_redundant  = -0.01,
        reward_step       = -0.005,   # presión para moverse y explorar
        seed          = 42,
    )

    base_env = MARLExploration3D(cfg)
    return base_env, dominant_id


def _rebuild_rollout_buffer(model: PPO) -> None:
    """
    Recrea el RolloutBuffer de SB3 con los n_steps actuales del modelo.
    Es necesario al cambiar de fase porque n_steps puede cambiar.
    """
    model.rollout_buffer = RolloutBuffer(
        buffer_size      = model.n_steps,
        observation_space= model.observation_space,
        action_space     = model.action_space,
        device           = model.device,
        gamma            = model.gamma,
        gae_lambda       = model.gae_lambda,
        n_envs           = model.n_envs,
    )


def _apply_lr_decay(model: PPO, factor: float = 0.8, min_lr: float = 5e-5) -> float:
    """
    Aplica decay al learning rate y actualiza también el optimizador de PyTorch,
    para que el nuevo lr sea efectivo inmediatamente en la siguiente actualización.
    """
    old_lr = float(model.learning_rate) if not callable(model.learning_rate) \
             else model.learning_rate(1.0)
    new_lr = max(old_lr * factor, min_lr)
    model.learning_rate = new_lr

    # Actualizar el optimizador de la política (PyTorch)
    if hasattr(model, "policy") and model.policy is not None:
        for pg in model.policy.optimizer.param_groups:
            pg["lr"] = new_lr

    return new_lr


def train_team_sb3(team_id: int, start_phase: int = 0):
    """
    Entrena el equipo `team_id` a través del curriculum completo.

    Parámetros
    ----------
    team_id     : ID del equipo (1–10), tal como está definido en TEAMS.
    start_phase : Índice de fase desde la que arrancar (0 = desde el principio).
                  Las fases anteriores a start_phase se saltan.
    """
    team           = TEAMS[team_id]
    dominant_id    = team.dominant.id

    print(f"\n{'='*70}")
    print(f"  EQUIPO {team_id}: {team.name}  {team.label}")
    print(f"  {team.description}")
    print(f"  Agentes: {team.n_agents}  |  Arquetipo dominante: {team.dominant.name}")
    print(f"{'='*70}\n")

    model: Optional[PPO] = None

    for phase in CURRICULUM:

        # ── Saltar fases anteriores a start_phase ────────────────
        if phase.idx < start_phase:
            print(f"  [SKIP] Fase {phase.idx}: {phase.name}  (< start_phase={start_phase})")
            continue

        # ── Saltar fases sin episodios de entrenamiento ──────────
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
        vec_env = SB3MultiAgentVecEnv(base_env, dominant_arch_id)

        ppo_params = _ppo_kwargs(phase.max_steps, team.n_agents)

        # ── Inicializar o actualizar el modelo ────────────────────
        if model is None:
            # Primera fase: crear el modelo desde cero
            model = PPO("MlpPolicy", vec_env, **ppo_params)
            print(f"  Modelo PPO creado. Parámetros: {ppo_params}")
        else:
            # Fases siguientes: transferir pesos, ajustar entorno y buffer
            model.set_env(vec_env)
            model.n_steps    = ppo_params["n_steps"]
            model.batch_size = ppo_params["batch_size"]
            _rebuild_rollout_buffer(model)

            # Decaer lr para no destruir lo aprendido en la nueva fase
            new_lr = _apply_lr_decay(model, factor=0.8, min_lr=5e-5)
            print(f"  Learning rate ajustado → {new_lr:.2e}")

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
        model.learn(
            total_timesteps     = total_timesteps,
            callback            = callback,
            reset_num_timesteps = False,   # mantiene el contador global entre fases
            log_interval        = 10,
        )

        # ── Checkpoint ───────────────────────────────────────────
        ckpt_name = f"{OUT_DIR}/model_team{team_id}_phase{phase.idx}"
        model.save(ckpt_name)
        print(f"✓ Fase {phase.idx} guardada: {ckpt_name}.zip\n")

        vec_env.close()

    print(f"\n{'='*70}")
    print(f"  Entrenamiento completo — Equipo {team_id}: {team.name}")
    print(f"  Checkpoints en: {OUT_DIR}/")
    print(f"{'='*70}\n")


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrenamiento PPO (SB3) — MARL Exploración 3D"
    )
    parser.add_argument(
        "--team",  type=int, default=1,
        help="ID del equipo a entrenar (1-10, default=1)"
    )
    parser.add_argument(
        "--fase",  type=int, default=0,
        help="Fase de inicio del curriculum (0-5, default=0)"
    )
    args = parser.parse_args()

    train_team_sb3(team_id=args.team, start_phase=args.fase)