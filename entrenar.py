from __future__ import annotations
import os, sys, time, json, argparse
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import gymnasium as gym  # SB3 usa Gymnasium nativamente

# Stable-Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback

sys.path.insert(0, ".")
from marl_exploration_3d import MARLExploration3D, ExplorationConfig
from entrenar_agentes import TEAMS, CURRICULUM, ARCHETYPES, AgentObsWrapper

OUT_DIR = "checkpoints_sb3"
os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# 1. WRAPPER PARA CONVERTIR MARL EN UN ENTORNO VECTORIZADO SB3
# ═══════════════════════════════════════════════════════════════
class SB3MultiAgentVecEnv(VecEnv):
    """
    Convierte un entorno MARL donde N agentes actúan simultáneamente
    en un entorno vectorizado de SB3 con N sub-entornos independientes.
    Esto implementa 'Parameter Sharing' nativo y ultra-rápido en SB3.
    """
    def __init__(self, env: MARLExploration3D, archetype_id: int):
        self.env = env
        self.arch = ARCHETYPES[archetype_id]
        self.n_agents = env.cfg.n_agents

        self.agent_indices = list(range(self.n_agents))

        # BUG 1 — self.at no existe: se usaba self.at antes de definirlo.
        # ARCHETYPES[archetype_id] ya está en self.arch; AgentObsWrapper espera
        # un AgentType, así que se pasa self.arch directamente.
        # ORIGINAL: self.wrapper = AgentObsWrapper(self.at, self.n_agents)
        self.wrapper = AgentObsWrapper(self.arch, self.n_agents)

        # BUG 2 — reset() llamado en __init__ sin seed ni opciones.
        # El entorno interno ya ha sido inicializado por el constructor de
        # MARLExploration3D, por lo que reset() aquí provoca un reset extra
        # silencioso que puede dejar self.info = {} (sin el campo 'step' ni
        # 'coverage_ratio') y desincroniza el estado interno del entorno con
        # el que SB3 recibirá en la primera llamada real a reset().
        # Solución: llamar a reset con seed fijo para obtener solo obs_dim,
        # después vaciar self.info para que sea evidente si se usa sin reset.
        obs_p, _ = self.env.reset(seed=env.cfg.seed)
        sample_raw = self.wrapper.process(obs_p[0], 0, self.env)
        obs_dim = sample_raw.shape[0]
        self.info: Dict = {}

        observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        action_space = gym.spaces.Discrete(self.env.n_actions)

        super().__init__(
            num_envs=self.n_agents,
            observation_space=observation_space,
            action_space=action_space,
        )
        self.actions_dict: Dict[int, int] = {}

    # BUG 3 — reset() no acepta **kwargs que SB3 pasa internamente
    # (seed=, options=).  VecEnv.reset() en SB3 ≥1.7 pasa estos argumentos
    # al envoltorio; sin ellos, Python lanza TypeError y el entrenamiento
    # nunca arranca.
    # ORIGINAL: def reset(self):
    def reset(self, **kwargs) -> np.ndarray:
        # Propagamos seed si SB3 la envía, para reproducibilidad.
        seed = kwargs.get("seed", None)
        obs_dict, self.info = self.env.reset(seed=seed) if seed is not None \
            else self.env.reset()
        self.wrapper.reset(self.env, self.agent_indices)

        for i in self.agent_indices:
            self.wrapper.update_private_map(i, self.env)

        return self._get_processed_obs(obs_dict)

    def _get_processed_obs(self, obs_dict) -> np.ndarray:
        obs_list = []
        for i in self.agent_indices:
            proc_obs = self.wrapper.process(obs_dict[i], i, self.env)
            obs_list.append(proc_obs)
        return np.array(obs_list, dtype=np.float32)

    def step_async(self, actions: np.ndarray) -> None:
        # BUG 4 — actions llegan como np.int64 pero el entorno espera int Python.
        # np.int64 no es idéntico a int en todos los contextos (e.g. al usarse
        # como índice de un dict o al compararse con Action(IntEnum)); hay que
        # convertir explícitamente.
        # ORIGINAL: self.actions_dict = {i: act for i, act in enumerate(actions)}
        self.actions_dict = {i: int(act) for i, act in enumerate(actions)}

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        obs_next, rewards, terminated, truncated, info = self.env.step(self.actions_dict)
        self.info = info

        done = terminated or truncated
        dones = np.array([done] * self.n_agents, dtype=bool)

        processed_rewards = []
        for i in self.agent_indices:
            extra_pen = self.wrapper.redundancy_penalty(i, self.env)
            processed_rewards.append(rewards[i] + extra_pen)
            self.wrapper.update_private_map(i, self.env)

        rewards_arr = np.array(processed_rewards, dtype=np.float32)

        # BUG 5 — obs se procesa ANTES del auto-reset cuando done=True.
        # Si el episodio termina, obs_next ya es la observación del estado
        # terminal, que no tiene sentido para la política.  SB3 espera recibir
        # la observación del NUEVO episodio (post-reset) en dones=True.
        # La llamada a reset() debe ocurrir ANTES de construir proc_obs.
        # ORIGINAL:
        #   proc_obs = self._get_processed_obs(obs_next)   ← obs terminal
        #   if done:
        #       proc_obs = self.reset()                    ← se sobreescribe bien,
        #                                                    pero _get_processed_obs
        #                                                    ya fue llamado en vano
        # Solución: primero auto-reset si procede, luego procesar obs.
        if done:
            obs_next_for_policy = self.reset()          # devuelve obs post-reset
        else:
            obs_next_for_policy = self._get_processed_obs(obs_next)

        # BUG 6 — infos no incluye "terminal_observation" cuando done=True.
        # SB3 necesita guardar la última observación del episodio para calcular
        # correctamente el valor bootstrap en el límite del episodio
        # (ver SB3 on_rollout_end).  Sin este campo, el valor estimado al final
        # del episodio es 0 en todos los casos (equivalente a truncated=False),
        # lo que sesga las ventajas GAE cuando se usa truncation.
        # ORIGINAL: infos = [info.copy() for _ in range(self.n_agents)]
        terminal_obs = self._get_processed_obs(obs_next) if done else None
        infos = []
        for i in range(self.n_agents):
            d = info.copy()
            if done and terminal_obs is not None:
                d["terminal_observation"] = terminal_obs[i]
            infos.append(d)

        return obs_next_for_policy, rewards_arr, dones, infos

    def close(self) -> None:
        self.env.close()

    def env_is_wrapped(self, wrapper_class, indices=None): return [False] * self.num_envs
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs): pass
    def get_attr(self, attr_name, indices=None): pass
    def set_attr(self, attr_name, value, indices=None): pass


# ═══════════════════════════════════════════════════════════════
# 2. CALLBACK PARA SEGUIMIENTO DE MÉTRICAS Y CURRICULUM ADVANCE
# ═══════════════════════════════════════════════════════════════
class MARLCurriculumCallback(BaseCallback):
    """Monitorea el coverage_ratio y detiene el entrenamiento de la fase si converge."""
    def __init__(self, advance_cov: float, advance_k: int, verbose=0):
        super().__init__(verbose)
        self.advance_cov = advance_cov
        self.advance_k = advance_k
        self.recent_coverages: List[float] = []

    def _on_step(self) -> bool:
        dones = self.locals["dones"]
        if dones[0]:
            # BUG 7 — acceso directo a training_env.info es frágil e incorrecto.
            # self.training_env es un VecEnv; su atributo .info no es parte de
            # la interfaz pública de VecEnv y no existe en todas las versiones de
            # SB3.  La forma correcta de leer info en un callback es a través de
            # self.locals["infos"], que SB3 garantiza que esté disponible en
            # _on_step() y contiene la lista de dicts devuelta por step_wait().
            # ORIGINAL: info = self.training_env.info
            infos = self.locals["infos"]
            # Tomamos el info del agente 0 (todos comparten el mismo episodio)
            info = infos[0] if infos else {}

            cov = info.get("coverage_ratio", 0.0)
            self.recent_coverages.append(cov)
            if len(self.recent_coverages) > self.advance_k:
                self.recent_coverages.pop(0)

            mean_cov = np.mean(self.recent_coverages)

            if self.verbose > 0:
                n = len(self.recent_coverages)
                print(
                    f" -> Episodio finalizado. "
                    f"Cobertura: {cov*100:.1f}% | "
                    f"Media reciente ({n} eps): {mean_cov*100:.1f}%"
                )

            if (len(self.recent_coverages) >= self.advance_k
                    and mean_cov >= self.advance_cov):
                print(
                    f"│ ⚡ Avance anticipado activado por SB3 "
                    f"(Media: {mean_cov*100:.1f}% >= "
                    f"Objetivo: {self.advance_cov*100:.1f}%)"
                )
                return False
        return True


# ═══════════════════════════════════════════════════════════════
# 3. BUCLE PRINCIPAL DE ENTRENAMIENTO (MIGRACIÓN v3.0)
# ═══════════════════════════════════════════════════════════════
def train_team_sb3(team_id: int, start_phase: int = 1, is_fast: bool = False):
    team = TEAMS[team_id]
    print(f"\nIniciando entrenamiento SB3 para Equipo {team.name}...")

    # BUG 8 — se usa team.composition[0] como arquetipo dominante, ignorando
    # la lógica de dominancia ya definida en TeamConfig.dominant.
    # Para equipos mixtos (ej. MIXTO-C [4,4,3,3]) el índice 0 da el primer
    # arquetipo por orden de composición, no el de mayor complejidad de comm.
    # TeamConfig.dominant() ya resuelve esto con la tabla de prioridad correcta.
    # ORIGINAL: dominant_arch_id = team.composition[0]
    dominant_arch_id = team.dominant.id

    model: Optional[PPO] = None

    for phase in CURRICULUM:
        if phase.idx < start_phase:
            continue
        if phase.n_episodes == 0:
            continue

        # BUG 9 — mutación directa de los objetos globales de CURRICULUM.
        # CURRICULUM es una lista global de dataclasses; modificar phase.grid_h
        # aquí afecta de forma permanente a todas las llamadas futuras (otros
        # equipos, llamadas sucesivas al script).  Hay que trabajar con una
        # copia local de los valores, no con el objeto compartido.
        # ORIGINAL:
        #   phase.grid_h = min(phase.grid_h, 25)
        #   phase.grid_w = min(phase.grid_w, 25)
        #   ...
        if is_fast:
            grid_h      = min(phase.grid_h,    25)
            grid_w      = min(phase.grid_w,    25)
            max_steps   = min(phase.max_steps, 500)
            n_episodes  = max(5, phase.n_episodes // 30)
        else:
            grid_h      = phase.grid_h
            grid_w      = phase.grid_w
            max_steps   = phase.max_steps
            n_episodes  = phase.n_episodes

        print(f"\n--- CONFIGURANDO FASE {phase.idx}: {phase.name} ({grid_h}x{grid_w}) ---")

        # 1. Instanciar entorno base
        rwall = 0.0 if phase.idx <= 1 else -0.01
        cfg = ExplorationConfig(
            grid_h=grid_h, grid_w=grid_w, n_floors=phase.n_floors,
            wall_density=phase.wall_density, n_stairs=phase.n_stairs,
            min_connected=min(
                phase.min_connected,
                int(grid_h * grid_w * phase.n_floors * 0.6)
            ),
            n_agents=team.n_agents,
            obs_radius=ARCHETYPES[dominant_arch_id].obs_radius,
            comm_mode=ARCHETYPES[dominant_arch_id].comm_mode,
            msg_dim=ARCHETYPES[dominant_arch_id].msg_dim,
            reward_mode=ARCHETYPES[dominant_arch_id].reward_mode,
            reward_alpha=ARCHETYPES[dominant_arch_id].reward_alpha,
            max_steps=max_steps,
            reward_wall=rwall,
            reward_new_cell=70.0,
            reward_completion=phase.completion_bonus,
            reward_redundant=-0.01,
            seed=42,
        )
        base_env = MARLExploration3D(cfg)

        # 2. Envolver en VecEnv para Stable-Baselines3
        vec_env = SB3MultiAgentVecEnv(base_env, dominant_arch_id)

        # 3. Inicializar o Transferir Pesos del Modelo PPO de SB3
        if model is None:
            model = PPO(
                "MlpPolicy",
                vec_env,
                learning_rate=5e-4,
                # BUG 10 — n_steps demasiado grande causa OOM y rollouts vacíos.
                # n_steps en SB3 es el número de pasos POR sub-entorno antes de
                # actualizar; con N agentes (sub-envs) el rollout real tiene
                # n_steps × N pasos.  Usar phase.max_steps directamente puede
                # ser de miles de pasos × N agentes, lo que agota RAM y hace
                # que el modelo nunca actualice hasta el final del episodio.
                # Un valor razonable alineado con entrenar_agentes.py es
                # max_steps // n_agents, pero acotado para no ser demasiado
                # pequeño (mínimo 64 pasos útiles por sub-env).
                # ORIGINAL: n_steps=phase.max_steps
                n_steps=max(64, max_steps // team.n_agents),
                batch_size=32,
                n_epochs=10,
                gamma=0.95,
                gae_lambda=0.90,
                clip_range=0.3,
                ent_coef=0.05,
                vf_coef=1.0,
                max_grad_norm=0.5,
                policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
                verbose=0,
                device="auto",
            )
        else:
            # BUG 11 — model.learning_rate es un callable en SB3 (puede ser un
            # schedule), no un float.  Leer y escribir model.learning_rate como
            # float falla si PPO fue instanciado con un schedule, y en cualquier
            # caso no actualiza el optimizador interno (optimizer.param_groups).
            # La única forma correcta de cambiar lr en caliente es redefinir el
            # schedule y reasignar model.learning_rate como callable, o acceder
            # directamente al optimizador.
            # ORIGINAL:
            #   model.set_env(vec_env)
            #   model.learning_rate = max(model.learning_rate * 0.8, 5e-5)
            model.set_env(vec_env)
            # Leer lr actual del optimizador (fuente de verdad real)
            current_lr = model.policy.optimizer.param_groups[0]["lr"]
            new_lr = max(current_lr * 0.8, 5e-5)
            # Actualizar tanto el schedule (para que SB3 no lo sobreescriba)
            # como el optimizador directamente
            model.learning_rate = new_lr
            for pg in model.policy.optimizer.param_groups:
                pg["lr"] = new_lr
            print(f"│  LR ajustada: {current_lr:.2e} → {new_lr:.2e}")

        total_timesteps = n_episodes * max_steps

        callback = MARLCurriculumCallback(
            advance_cov=phase.advance_cov,
            advance_k=phase.advance_k,
            verbose=1,
        )

        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=False,
        )

        ckpt_name = f"{OUT_DIR}/model_team_{team_id}_phase_{phase.idx}"
        model.save(ckpt_name)
        print(f"✓ Fase {phase.idx} guardada con éxito en {ckpt_name}.zip")

        vec_env.close()


if __name__ == "__main__":
    # Prueba rápida de entrenamiento con SB3: Equipo 1, Fase 1, Modo rápido.
    train_team_sb3(team_id=1, start_phase=1, is_fast=True)