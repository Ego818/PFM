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
"""
entrenar.py  —  Entrenamiento PPO multi-política con Stable-Baselines3
=======================================================================
Entrena un equipo del MARL de exploración 3D usando PPO de SB3
con curriculum learning progresivo, respetando la composición real
de arquetipos definida en agente_info.py.

ARQUITECTURA MULTI-POLÍTICA (v3)
=================================
Cada arquetipo único del equipo tiene su propio modelo PPO independiente.
Los agentes que comparten arquetipo comparten parámetros (parameter sharing).

Ejemplo — Equipo 8 MIXTO-C [4,4,3,3]:
  · modelo_arch4  →  agentes 0, 1  (CARTÓGRAFO: obs_radius=8, private_map)
  · modelo_arch3  →  agentes 2, 3  (MENSAJERO: msg_dim=8, comm_mode=explicit)

El entorno se construye con los parámetros del arquetipo dominante (el de mayor
prioridad) para que el ExplorationConfig sea coherente. Las observaciones de
cada agente se enrutan al modelo de su arquetipo y cada modelo actúa solo para
sus agentes asignados.

CLASES PRINCIPALES
==================
  ArchetypeVecEnv          VecEnv de SB3 para un subconjunto de agentes
                           (los que pertenecen a un arquetipo concreto).
                           Comparte el entorno base con los demás arquetipos.

  MultiPolicyTrainer       Orquesta N modelos PPO (uno por arquetipo único),
                           hace rollouts conjuntos paso a paso y llama a
                           learn() en cada modelo con sus propios datos.

COMPATIBILIDAD CON video.py
============================
Los checkpoints se guardan con el mismo esquema que antes:
  checkpoints_sb3/model_team{T}_phase{P}_arch{A}.zip
video.py ya itera sobre todos los .zip disponibles, así que no requiere cambios.
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

class ArchetypeVecEnv(VecEnv):
    """
    VecEnv de SB3 para un subconjunto de agentes que comparten arquetipo.

    Expone SOLO los agentes de `agent_indices` como sub-entornos virtuales.
    El entorno base (MARLExploration3D) es compartido con los demás
    ArchetypeVecEnv del mismo equipo; el paso real del entorno lo gestiona
    MultiPolicyTrainer, no este wrapper.

    Flujo por paso
    ──────────────
    1. MultiPolicyTrainer llama a collect_actions(obs_dict) → acciones para
       los agentes de este arquetipo.
    2. MultiPolicyTrainer ejecuta env.step(all_actions) con las acciones de
       todos los arquetipos combinadas.
    3. MultiPolicyTrainer llama a ingest_step(obs_next, rewards, done, info)
       para que este wrapper actualice su buffer interno.
    4. Cuando SB3 llama a step_async/step_wait, se sirven los datos ya
       almacenados (no se vuelve a llamar al entorno).

    NORMALIZACIÓN DE RECOMPENSAS
    ─────────────────────────────
    Welford online independiente por arquetipo. Cada grupo de agentes
    normaliza sus propias recompensas, lo que es correcto porque los
    arquetipos pueden tener escalas de recompensa distintas (alpha diferente).
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
        self.agent_indices = agent_indices   # índices globales de este arquetipo
        self.info: Dict    = {}

        obs_dim = obs_dict_init[agent_indices[0]].shape

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

    # ── Extracción de obs para este subconjunto ──────────────────

    def _extract_obs(self, obs_dict: Dict[int, np.ndarray]) -> np.ndarray:
        return np.array(
            [obs_dict[i] for i in self.agent_indices], dtype=np.float32
        )

    # ── API llamada por MultiPolicyTrainer ───────────────────────

    def get_obs(self) -> np.ndarray:
        """Devuelve la observación actual del subconjunto de agentes."""
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
        obs_next: Dict[int, np.ndarray],
        rewards:  Dict[int, float],
        done:     bool,
        info:     Dict,
    ) -> None:
        """
        Deposita en el buffer interno los resultados del paso del entorno.
        Llamado por MultiPolicyTrainer tras env.step().
        """
        self.info = info
        obs_arr   = self._extract_obs(obs_next)
        rews_raw  = np.array([rewards[i] for i in self.agent_indices], dtype=np.float32)
        rews_arr  = self._normalize_rewards(rews_raw)
        dones_arr = np.array([done] * len(self.agent_indices), dtype=bool)
        infos     = [info.copy() for _ in self.agent_indices]

        if done:
            for idx in range(len(self.agent_indices)):
                infos[idx]["terminal_observation"] = obs_arr[idx].copy()
                infos[idx]["TimeLimit.truncated"]  = bool(info.get("TimeLimit.truncated", False))
            # obs post-reset ya viene en obs_next (el trainer hace reset antes de llamar ingest)
        self._pending_obs   = obs_arr
        self._pending_rews  = rews_arr
        self._pending_dones = dones_arr
        self._pending_infos = infos

    # ── API VecEnv requerida por SB3 ────────────────────────────

    def reset(self, **kwargs) -> np.ndarray:
        # El reset real lo gestiona MultiPolicyTrainer; aquí devolvemos
        # la obs que ya está en el buffer (post-reset inyectada por ingest_step).
        assert self._pending_obs is not None
        return self._pending_obs

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = {
            global_i: int(a)
            for global_i, a in zip(self.agent_indices, actions)
        }

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        # Los datos ya fueron inyectados por ingest_step; simplemente los devolvemos.
        assert self._pending_rews  is not None, "ingest_step no llamado antes de step_wait"
        obs   = self._pending_obs
        rews  = self._pending_rews
        dones = self._pending_dones
        infos = self._pending_infos
        # Limpiar para detectar uso indebido
        self._pending_rews  = None
        self._pending_dones = None
        self._pending_infos = None
        return obs, rews, dones, infos

    # ── Métodos abstractos restantes ─────────────────────────────

    def close(self) -> None:
        pass   # el entorno base lo cierra MultiPolicyTrainer

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


# Mantener alias para compatibilidad con video.py
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
      4. SB3 recoge los datos a través de la API VecEnv estándar.

    El entrenamiento (model.learn) se delega a cada PPO por separado,
    pero los rollouts están sincronizados en el mismo entorno compartido.
    """

    def __init__(
        self,
        base_env:    MARLExploration3D,
        team_id:     int,
        ppo_params:  dict,
    ):
        from stable_baselines3 import PPO as _PPO

        self.env     = base_env
        self.team    = TEAMS[team_id]
        self.composition = self.team.composition   # [1,1,3,3] etc.

        # Mapear arquetipo → lista de índices globales de agentes
        self.arch_to_agents: Dict[int, List[int]] = {}
        for agent_idx, arch_id in enumerate(self.composition):
            self.arch_to_agents.setdefault(arch_id, []).append(agent_idx)

        # Reset inicial para obtener obs de referencia
        obs_dict, _ = self.env.reset()

        # Crear un ArchetypeVecEnv + PPO por cada arquetipo único
        self.vec_envs: Dict[int, ArchetypeVecEnv] = {}
        self.models:   Dict[int, _PPO]             = {}

        for arch_id, agent_indices in self.arch_to_agents.items():
            vec_env = ArchetypeVecEnv(
                env           = self.env,
                archetype_id  = arch_id,
                agent_indices = agent_indices,
                obs_dict_init = obs_dict,
            )
            model = _PPO("MlpPolicy", vec_env, **ppo_params)
            self.vec_envs[arch_id] = vec_env
            self.models[arch_id]   = model

        # obs actual del entorno (compartida entre todos los VecEnv)
        self._current_obs: Dict[int, np.ndarray] = obs_dict

    # ── Actualizar entorno al cambiar de fase ────────────────────

    def set_phase_env(self, base_env: MARLExploration3D, ppo_params: dict) -> None:
        """
        Sustituye el entorno base al cambiar de fase del curriculum.
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
            model = self.models[arch_id]
            model.set_env(new_vec)
            model.n_steps    = ppo_params["n_steps"]
            model.batch_size = ppo_params["batch_size"]
            _rebuild_rollout_buffer(model)
            new_lr = _apply_lr_decay(model, factor=0.8, min_lr=5e-5)
            print(f"    Arquetipo {arch_id} ({ARCHETYPES[arch_id].name}): lr → {new_lr:.2e}")

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

        # 3. Si terminó, hacer reset y usar la obs post-reset
        if done:
            obs_next, _ = self.env.reset()

        self._current_obs = obs_next

        # 4. Distribuir resultados a cada ArchetypeVecEnv
        for arch_id, vec_env in self.vec_envs.items():
            vec_env.ingest_step(obs_next, rewards, done, info)

        return done, info

    # ── Entrenamiento de todos los modelos ───────────────────────

    def learn(self, total_timesteps: int, callback: "BaseCallback") -> None:
        """
        Ejecuta el entrenamiento sincronizado. En cada paso, todos los
        modelos avanzan juntos un tick del entorno. SB3 gestiona sus
        propios rollout buffers internamente.

        Dado que SB3 espera llamar a env.step() a través de su VecEnv,
        interceptamos el proceso usando collect_rollouts personalizado.
        Para mantener la compatibilidad con SB3 usamos el truco de hacer
        que cada modelo llame a learn() en su VecEnv ya "pre-alimentado"
        por step_all().
        """
        # Ejecutar steps manuales y alimentar los VecEnv
        steps_done = 0
        ep_done    = False

        while steps_done < total_timesteps:
            ep_done, info = self.step_all()
            steps_done += 1

            # Disparar collect_rollouts de SB3 cuando se llena el buffer
            # Para ello dejamos que cada modelo haga un mini-learn de 1 step
            # acumulando en su buffer. Cuando el buffer se llena, SB3 actualiza.
            for arch_id, model in self.models.items():
                vec_env = self.vec_envs[arch_id]
                # Simular que SB3 consume el step: llamamos step_async/step_wait
                # con las acciones que ya tomamos en step_all
                vec_env.step_async(
                    np.array(
                        [vec_env._actions.get(i, 0) for i in vec_env.agent_indices],
                        dtype=np.int64,
                    )
                )

            # Criterio de avance: reutilizamos el callback
            if ep_done:
                cov = float(info.get("coverage_ratio", 0.0))
                if hasattr(callback, "recent_covs"):
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

        # Actualizar los modelos con los datos acumulados en sus buffers
        # Llamamos learn con 0 steps adicionales para forzar la actualización
        # de los pesos usando los datos ya recolectados.
        for arch_id, model in self.models.items():
            model.learn(
                total_timesteps     = total_timesteps,
                callback            = None,
                reset_num_timesteps = False,
                log_interval        = 10,
            )

    def save(self, base_path: str) -> None:
        """Guarda un checkpoint por arquetipo."""
        for arch_id, model in self.models.items():
            path = f"{base_path}_arch{arch_id}"
            model.save(path)
            print(f"    ✓ Arquetipo {arch_id} ({ARCHETYPES[arch_id].name}): {path}.zip")

    def close(self) -> None:
        self.env.close()


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