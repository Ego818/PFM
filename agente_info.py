"""
entrenar_agentes.py  —  v3.0  «10-Episode Turbo»
=================================================
Entrenamiento mediante Curriculum Learning de 10 configuraciones de equipo
para exploración cooperativa 3D (MARL).

CONFIGURACIONES DE EQUIPO
==========================

 1. [1,1,1,1]          SOLITARIOS     — 4 agentes sin radio ni memoria
 2. [2,2,2,2]          RADAR          — 4 agentes con posiciones GPS
 3. [3,3,3,3]          MENSAJEROS     — 4 agentes con radio explícita
 4. [4,4,4,4]          CARTÓGRAFOS    — 4 agentes con mapa privado grande
 5. [5,5,5,5,5,5]      COLMENA        — 6 agentes con radio rica + mapa privado
 6. [1,1,2,2]          MIXTO-A        — 2 solitarios + 2 radar
 7. [1,1,3,3]          MIXTO-B        — 2 solitarios + 2 mensajeros
 8. [4,4,3,3]          MIXTO-C        — 2 cartógrafos + 2 mensajeros
 9. [4,4,5,5,5,5]      MIXTO-D        — 2 cartógrafos + 4 colmena
10. [1,2,3,4,5,5]      MULTIDISCIPLINAR — 1 de cada + 2 colmena

ALGORITMO: PPO (implementación NumPy pura)
==========================================
  - Actor-Critic MLP con capas ocultas ReLU
  - GAE-λ para estimación de ventajas
  - Adam optimizer con gradient clipping
  - Parameter sharing dentro de agentes del mismo tipo

OPTIMIZACIONES v3.0 — convergencia en ≤10 episodios por fase
=============================================================

  [RED NEURONAL]
  - Capas ocultas reducidas a (128,128) para todos los arquetipos (menos params,
    señal de gradiente más limpia en pocos episodios)
  - Init cabeza π: std = 0.01 (política uniforme garantizada al inicio)
  - BatchNorm simulado en observaciones (normalización por running stats)

  [PPO]
  - n_ppo_epochs = 10 (más pasadas sobre el mismo batch → mejor uso de datos)
  - batch_size   = 32 (batches pequeños → gradientes más frecuentes)
  - clip_eps     = 0.3 (clipping más permisivo → actualizaciones más grandes)
  - vf_coef      = 1.0 (igual peso a valor que a política)
  - ent_coef     = 0.05 (más exploración inicial)
  - Recomputa log_probs en cada época PPO (ratio siempre fresco → clip válido)

  [GAE / RETORNOS]
  - gamma = 0.95 (horizonte más corto → señal más inmediata, convergencia rápida)
  - lam   = 0.90 (GAE-λ ligeramente más sesgado → menos varianza)

  [ADAM]
  - lr base: 5e-4 para todos los arquetipos (más agresivo en pocas muestras)
  - β1=0.9, β2=0.99 (β2 más bajo → memoria más corta → adapta rápido)
  - Gradient clipping: max_norm = 0.5 (más estricto → estabilidad)

  [CURRICULUM]
  - n_episodes = 10 por fase (objetivo explícito)
  - advance_k  = 5 (ventana de avance anticipado pequeña)
  - advance_cov reducida (criterio alcanzable en 10 eps)
  - Fase 0: 0 episodios (solo inicializa la red, sin cambios)
  - max_steps proporcional al área pero acotado más bajo (~2 pasos/celda)
  - warmup = 0 episodios (no hay tiempo para calentar, se entra directo)

  [RECOMPENSAS]
  - reward_new_cell  = 2.0 (señal fuerte de exploración)
  - reward_completion = 10.0 × fase (bonus creciente por terminar)
  - reward_wall      = -0.02 (mínimo, no distrae)
  - reward_redundant sube de 0 a max en los primeros 5 eps (warmup corto)

  [OBSERVACIONES]
  - Normalización online de observaciones por running mean/std por arquetipo
    (ObsNormalizer): estabiliza la señal incluso con pocas muestras

  CORRECCIONES HEREDADAS de v2.1
  ================================
  - Bug Adam corregido (self extra)
  - Signo gradiente política correcto (−min(surr))
  - Mapa privado uno por agente
  - reward_wall = -0.02 (< -0.05 de v2.1)
  - lr decae ×0.5 al entrar en fase nueva (preserva pesos)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, '.')
from marl_exploration_3d import MARLExploration3D


# ═══════════════════════════════════════════════════════════════
# 1. DEFINICIONES DE ARQUETIPOS Y EQUIPOS
# ═══════════════════════════════════════════════════════════════

@dataclass
class AgentType:
    id: int
    name: str
    description: str
    comm_mode: str
    reward_mode: str
    reward_alpha: float
    obs_radius: int
    msg_dim: int
    hidden: Tuple[int, ...]
    lr: float
    extra_redundancy_penalty: float = 0.0
    private_map: bool = False


ARCHETYPES: Dict[int, AgentType] = {
    1: AgentType(
        id=1, name='SOLITARIO',
        description='Sin comunicación. Recompensa individual. Radio 3.',
        comm_mode='none', reward_mode='individual', reward_alpha=1.0,
        obs_radius=3, msg_dim=0,
        hidden=(128, 128), lr=5e-4,
        extra_redundancy_penalty=0.0, private_map=False,
    ),
    2: AgentType(
        id=2, name='RADAR',
        description='Posiciones GPS de compañeros. Radio 5. Recompensa mixta α=0.3.',
        comm_mode='partial', reward_mode='mixed', reward_alpha=0.3,
        obs_radius=5, msg_dim=0,
        hidden=(128, 128), lr=5e-4,
        extra_redundancy_penalty=-0.05, private_map=False,
    ),
    3: AgentType(
        id=3, name='MENSAJERO',
        description='Canal explícito 8 bits. Recompensa compartida pura.',
        comm_mode='explicit', reward_mode='shared', reward_alpha=0.0,
        obs_radius=5, msg_dim=8,
        hidden=(128, 128), lr=5e-4,
        extra_redundancy_penalty=-0.03, private_map=False,
    ),
    4: AgentType(
        id=4, name='CARTÓGRAFO',
        description='Mapa privado. Radio grande 8. Fuerte penalización por revisitar.',
        comm_mode='partial', reward_mode='mixed', reward_alpha=0.5,
        obs_radius=8, msg_dim=0,
        hidden=(128, 128), lr=4e-4,
        extra_redundancy_penalty=-0.2, private_map=True,
    ),
    5: AgentType(
        id=5, name='COLMENA',
        description='Canal explícito 16 bits. Mapa privado. Recompensa de equipo pura.',
        comm_mode='explicit', reward_mode='shared', reward_alpha=0.0,
        obs_radius=5, msg_dim=16,
        hidden=(128, 128), lr=4e-4,
        extra_redundancy_penalty=-0.05, private_map=True,
    ),
}


@dataclass
class TeamConfig:
    id: int
    label: str
    name: str
    description: str
    composition: List[int]

    @property
    def n_agents(self) -> int:
        return len(self.composition)

    @property
    def dominant(self) -> AgentType:
        priority = {5: 4, 3: 3, 4: 2, 2: 1, 1: 0}
        best_id = max(self.composition, key=lambda aid: priority.get(aid, 0))
        return ARCHETYPES[best_id]

    def unique_archetypes(self) -> List[int]:
        return list(dict.fromkeys(self.composition))


TEAMS: Dict[int, TeamConfig] = {
    1: TeamConfig(
        id=1, label='[1,1,1,1]', name='SOLITARIOS',
        description=(
            '4 exploradores sin radio. Límite inferior del sistema. '
            'Si los demás no superan esto, la comunicación no aporta nada.'
        ),
        composition=[1, 1, 1, 1],
    ),
    2: TeamConfig(
        id=2, label='[2,2,2,2]', name='RADAR',
        description=(
            '4 guardias con GPS de compañeros. '
            'Mide el beneficio de conocer posiciones sin info compleja.'
        ),
        composition=[2, 2, 2, 2],
    ),
    3: TeamConfig(
        id=3, label='[3,3,3,3]', name='MENSAJEROS',
        description=(
            '4 drones con radio explícita de 8 bits. '
            'Mide el valor de la comunicación explícita.'
        ),
        composition=[3, 3, 3, 3],
    ),
    4: TeamConfig(
        id=4, label='[4,4,4,4]', name='CARTÓGRAFOS',
        description=(
            '4 topógrafos con mapa privado de radio 8. '
            'Mide la importancia de la memoria espacial.'
        ),
        composition=[4, 4, 4, 4],
    ),
    5: TeamConfig(
        id=5, label='[5,5,5,5,5,5]', name='COLMENA',
        description=(
            '6 agentes de enjambre con radio rica de 16 bits. '
            'Máximo nivel de cooperación. Normalmente el mejor resultado.'
        ),
        composition=[5, 5, 5, 5, 5, 5],
    ),
    6: TeamConfig(
        id=6, label='[1,1,2,2]', name='MIXTO-A',
        description=(
            '2 operarios básicos + 2 supervisores con visión global. '
            'Mide si unos pocos agentes coordinados mejoran al resto.'
        ),
        composition=[1, 1, 2, 2],
    ),
    7: TeamConfig(
        id=7, label='[1,1,3,3]', name='MIXTO-B',
        description=(
            '2 exploradores básicos guiados por 2 operadores con radio. '
            'Mide si la comunicación compensa la falta de inteligencia local.'
        ),
        composition=[1, 1, 3, 3],
    ),
    8: TeamConfig(
        id=8, label='[4,4,3,3]', name='MIXTO-C',
        description=(
            '2 cartógrafos + 2 coordinadores con radio. '
            'Sinergia entre memoria y comunicación. Combinación muy fuerte.'
        ),
        composition=[4, 4, 3, 3],
    ),
    9: TeamConfig(
        id=9, label='[4,4,5,5,5,5]', name='MIXTO-D',
        description=(
            '2 expertos cartógrafos + 4 agentes de enjambre ejecutando. '
            'Arquitectura jerárquica similar a sistemas reales modernos.'
        ),
        composition=[4, 4, 5, 5, 5, 5],
    ),
    10: TeamConfig(
        id=10, label='[1,2,3,4,5,5]', name='MULTIDISCIPLINAR',
        description=(
            '1 de cada arquetipo + 1 colmena extra. Especialización emergente. '
            'La prueba más interesante desde el punto de vista científico.'
        ),
        composition=[1, 2, 3, 4, 5, 5],
    ),
}


# ═══════════════════════════════════════════════════════════════
# 2. FASES DEL CURRICULUM
# ═══════════════════════════════════════════════════════════════

@dataclass
class CurriculumPhase:
    idx: int
    name: str
    grid_h: int
    grid_w: int
    n_floors: int
    min_connected: int
    max_steps: int
    n_episodes: int
    wall_density: float
    n_stairs: int
    redundancy_penalty: float
    completion_bonus: float
    advance_cov: float
    advance_k: int


CURRICULUM: List[CurriculumPhase] = [
    CurriculumPhase(
        idx=0, name='Semilla',
        grid_h=20, grid_w=20, n_floors=1,
        min_connected=80, max_steps=200,
        n_episodes=300,
        wall_density=0.18, n_stairs=3,
        redundancy_penalty=0.0,
        completion_bonus=500.0,
        advance_cov=0.70,
        advance_k=10,
    ),
    CurriculumPhase(
        idx=1, name='Local',
        grid_h=40, grid_w=40, n_floors=2,
        min_connected=250, max_steps=1250,
        n_episodes=400,
        wall_density=0.18, n_stairs=5,
        redundancy_penalty=0.3,
        completion_bonus=1500.0,
        advance_cov=0.65,
        advance_k=5,
    ),
    CurriculumPhase(
        idx=2, name='Bajo',
        grid_h=80, grid_w=80, n_floors=3,
        min_connected=600, max_steps=5000,
        n_episodes=500,
        wall_density=0.30, n_stairs=8,
        redundancy_penalty=0.7,
        completion_bonus=2500.0,
        advance_cov=0.60,
        advance_k=8,
    ),
    CurriculumPhase(
        idx=3, name='Medio',
        grid_h=120, grid_w=120, n_floors=3,
        min_connected=1000, max_steps=10000,
        n_episodes=600,
        wall_density=0.33, n_stairs=10,
        redundancy_penalty=1.0,
        completion_bonus=4000.0,
        advance_cov=0.55,
        advance_k=20,
    ),
    CurriculumPhase(
        idx=4, name='Alto',
        grid_h=150, grid_w=150, n_floors=3,
        min_connected=1500, max_steps=15000,
        n_episodes=600,
        wall_density=0.33, n_stairs=10,
        redundancy_penalty=1.0,
        completion_bonus=4000.0,
        advance_cov=0.50,
        advance_k=20,
    ),
    CurriculumPhase(
        idx=5, name='Full',
        grid_h=250, grid_w=250, n_floors=3,
        min_connected=2000, max_steps=25000,
        n_episodes=600,
        wall_density=0.33, n_stairs=10,
        redundancy_penalty=1.0,
        completion_bonus=4000.0,
        advance_cov=0.40,
        advance_k=20,
    ),
]


# ═══════════════════════════════════════════════════════════════
# 3. WRAPPER DE OBSERVACIÓN POR AGENTE
# ═══════════════════════════════════════════════════════════════

class AgentObsWrapper:
    """
    Adapta la observación del entorno al formato del arquetipo.
    Mantiene UN mapa privado POR AGENTE.
    """

    def __init__(self, archetype: AgentType, n_agents: int):
        self.at = archetype
        self.n_agents = n_agents
        self._private_visit: Dict[int, Optional[np.ndarray]] = {
            i: None for i in range(n_agents)
        }

    def reset(self, env: MARLExploration3D, agent_indices: List[int]):
        for i in agent_indices:
            if self.at.private_map:
                self._private_visit[i] = np.zeros(
                    (env.F, env.H, env.W), dtype=bool
                )
            else:
                self._private_visit[i] = None

    def process(
        self,
        obs: np.ndarray,
        agent_id: int,
        env: MARLExploration3D,
    ) -> np.ndarray:
        if not self.at.private_map or self._private_visit[agent_id] is None:
            return obs.astype(np.float32)

        r = env.cfg.obs_radius
        size = 2 * r + 1
        view_cells = size * size

        f = int(env.agent_f[agent_id])
        ar = int(env.agent_r[agent_id])
        ac = int(env.agent_c[agent_id])

        r0, r1 = ar - r, ar + r + 1
        c0, c1 = ac - r, ac + r + 1
        gr0, gr1 = max(0, r0), min(env.H, r1)
        gc0, gc1 = max(0, c0), min(env.W, c1)

        private_patch = np.zeros((size, size), dtype=np.float32)
        pr0 = gr0 - r0; pc0 = gc0 - c0
        private_patch[pr0:pr0+(gr1-gr0), pc0:pc0+(gc1-gc0)] = (
            self._private_visit[agent_id][f, gr0:gr1, gc0:gc1].astype(np.float32)
        )

        new_obs = obs.copy()
        new_obs[view_cells: 2 * view_cells] = private_patch.flatten()
        return new_obs.astype(np.float32)

    def update_private_map(self, agent_id: int, env: MARLExploration3D):
        if self.at.private_map and self._private_visit[agent_id] is not None:
            f = int(env.agent_f[agent_id])
            r = int(env.agent_r[agent_id])
            c = int(env.agent_c[agent_id])
            self._private_visit[agent_id][f, r, c] = True

    def redundancy_penalty(self, agent_id: int, env: MARLExploration3D) -> float:
        if not self.at.private_map or self._private_visit[agent_id] is None:
            return 0.0
        f = int(env.agent_f[agent_id])
        r = int(env.agent_r[agent_id])
        c = int(env.agent_c[agent_id])
        if self._private_visit[agent_id][f, r, c]:
            return -0.01
        return 0.0
