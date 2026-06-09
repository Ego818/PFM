"""
marl_exploration_3d.py
======================
Entorno MARL para exploración cooperativa de un mapa 3D (250×250×3 plantas).

OBJETIVO
--------
  N agentes deben explorar JUNTOS la mayor fracción posible del mapa
  en el menor número de pasos. La tarea termina cuando se cubre el
  100 % de las celdas libres o se agota el presupuesto de pasos.

DISEÑO DEL ENTORNO
------------------
  Mapa : 250×250×3 plantas, generado proceduralmente en cada reset,
         garantizando ≥1500 celdas libres conexas.
         Paredes, escaleras (subir/bajar) varían en cada episodio.

  Exploración :
    - Cada celda libre tiene un flag "visitada" (False al inicio).
    - Al entrar un agente, la celda se marca visitada para todos.
    - La recompensa de equipo se basa en celdas nuevas descubiertas.

ESQUEMAS DE RECOMPENSA (configurables)
---------------------------------------
  "individual"  rᵢ = +1 por cada celda nueva que descubre el agente i
                     -0.01 por paso  -0.3 por colisión con pared
  "shared"      rᵢ = recompensa global / n_agents en cada paso
  "mixed"       rᵢ = α·rᵢ_individual + (1-α)·r_global/n_agents

COMUNICACIÓN (configurables)
-----------------------------
  "none"        obs local solo
  "partial"     obs local + posiciones de agentes dentro de comm_range
  "explicit"    idem + canal de mensaje continuo (msg_dim floats)

OBSERVACIÓN POR AGENTE (vector 1-D, CTDE-ready)
-------------------------------------------------
  [ mapa_local_celda    : (2r+1)² floats   — tipo de celda
    mapa_local_visita   : (2r+1)² floats   — ya visitada o no
    self_feats          : 5 floats          — planta, r, c, carry, step_norm
    otros_agentes       : N×5 floats        — en rango: df, dr, dc, visitadas_norm, en_rango
    mensaje_recibido    : msg_dim floats    — (solo explicit)
  ]

ESTADO GLOBAL (para critic centralizado en QMIX / MAPPO)
----------------------------------------------------------
  [ posicion_normalizada×N  |  visitadas_ratio  |  step_norm ]
  Dimensión: N×3 + 2  (muy compacto, sin serializar el grid)

ACCIONES (Discrete 7)
----------------------
  0 Arriba   1 Abajo   2 Izquierda   3 Derecha
  4 Subir escalera     5 Bajar escalera
  6 Enviar mensaje (solo comm_mode="explicit")

MÉTRICAS DE EPISODIO
--------------------
  coverage_ratio      : fracción de celdas libres visitadas [0,1]
  steps_to_50/75/100  : pasos para alcanzar hitos de cobertura
  redundancy_ratio    : fracción de pasos gastados en celdas ya vistas
  team_spread         : distancia media entre agentes (coordinación)
"""

from __future__ import annotations

import time
from collections import deque, Counter
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.ndimage import convolve


# ============================================================
# API mínima tipo Gymnasium (sin dependencia externa)
# ============================================================

class Space:
    def sample(self): raise NotImplementedError
    def contains(self, x) -> bool: raise NotImplementedError

class Box(Space):
    def __init__(self, low, high, shape, dtype=np.float32):
        self.low = np.full(shape, low, dtype=dtype)
        self.high = np.full(shape, high, dtype=dtype)
        self.shape = shape
        self.dtype = dtype
    def sample(self):
        return np.random.uniform(self.low, self.high).astype(self.dtype)
    def contains(self, x):
        x = np.asarray(x)
        return x.shape == self.shape and np.all(x >= self.low) and np.all(x <= self.high)
    def __repr__(self):
        return f"Box(shape={self.shape}, dtype={self.dtype})"

class Discrete(Space):
    def __init__(self, n):
        self.n = n
    def sample(self):
        return int(np.random.randint(0, self.n))
    def contains(self, x):
        return isinstance(x, (int, np.integer)) and 0 <= int(x) < self.n
    def __repr__(self):
        return f"Discrete({self.n})"

class Env:
    observation_space: Space = None
    action_space: Space = None
    metadata: dict = {}
    render_mode: Optional[str] = None
    def reset(self, *, seed=None, options=None): raise NotImplementedError
    def step(self, action): raise NotImplementedError
    def render(self): pass
    def close(self): pass


# ============================================================
# Constantes
# ============================================================

# Tipos de celda
LIBRE          = 0
PARED          = 1
ESCALERA_SUBIR = 2
ESCALERA_BAJAR = 3
N_CELL_TYPES   = 4

# Valores de observación para cada tipo
_CELL_OBS = np.array([0.0, 1.0, 0.5, 0.75], dtype=np.float32)

# Acciones
class Action(IntEnum):
    ARRIBA        = 0
    ABAJO         = 1
    IZQUIERDA     = 2
    DERECHA       = 3
    ESC_SUBIR     = 4
    ESC_BAJAR     = 5
    SEND_MSG      = 6

N_ACTIONS = len(Action)

_DR = np.array([-1,  1,  0, 0, 0, 0, 0], dtype=np.int32)
_DC = np.array([ 0,  0, -1, 1, 0, 0, 0], dtype=np.int32)


# ============================================================
# Configuración
# ============================================================

@dataclass
class ExplorationConfig:
    # Mapa
    grid_h: int            = 250
    grid_w: int            = 250
    n_floors: int          = 3
    wall_density: float    = 0.35
    n_stairs: int          = 12
    min_connected: int     = 1500

    # Agentes
    n_agents: int          = 6
    obs_radius: int        = 5       # vista local (2r+1)×(2r+1)

    # Comunicación
    comm_mode: str         = "none"  # "none" | "partial" | "explicit"
    comm_range: int        = 40      # distancia Manhattan máx
    msg_dim: int           = 8

    # Recompensa
    reward_mode: str       = "mixed"
    reward_alpha: float    = 0.4     # peso individual en mixed
    reward_new_cell: float = 1.0     # por celda nueva descubierta
    reward_step: float     = -0.01   # penalización por paso
    reward_wall: float     = -0.3    # penalización por colisión
    reward_redundant: float= -0.02   # penalización por pisar celda ya vista
    reward_completion: float= 50.0   # bonus al llegar al 100% de cobertura

    # Episodio
    max_steps: int         = 5000
    coverage_target: float = 1.0     # fracción objetivo (1.0 = 100%)

    # Semilla
    seed: Optional[int]    = None


# ============================================================
# Generador de mapas (igual que env_3d.py, standalone)
# ============================================================

class _MapGenerator:
    def __init__(self, cfg: ExplorationConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

    def generate(self):
        floors = self._gen_walls()
        stairs_up, stairs_down = self._place_stairs(floors)
        floors, stairs_up, stairs_down = self._ensure_connectivity(
            floors, stairs_up, stairs_down
        )
        return floors, stairs_up, stairs_down

    def _gen_walls(self):
        H, W, F = self.cfg.grid_h, self.cfg.grid_w, self.cfg.n_floors
        floors = np.zeros((F, H, W), dtype=np.int32)
        kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.float32)
        for f in range(F):
            g = (self.rng.random((H, W)) < self.cfg.wall_density).astype(np.int32)
            g[0,:] = g[-1,:] = g[:,0] = g[:,-1] = 1
            for _ in range(4):
                nb = convolve(g.astype(np.float32), kernel, mode='constant', cval=1.0)
                g = np.where(nb >= 5, 1, g)
                g = np.where((g == 1) & (nb == 0), 0, g)
            g[0,:] = g[-1,:] = g[:,0] = g[:,-1] = 1
            floors[f] = g
        return floors

    def _place_stairs(self, floors):
        H, W, F = self.cfg.grid_h, self.cfg.grid_w, self.cfg.n_floors
        n_st = self.cfg.n_stairs
        stairs_up, stairs_down = [], []
        for f in range(F - 1):
            free_f  = set(map(tuple, np.argwhere(floors[f]   == LIBRE)))
            free_f1 = set(map(tuple, np.argwhere(floors[f+1] == LIBRE)))
            cands   = list(free_f & free_f1)
            if len(cands) < n_st:
                extra = list(free_f)[:n_st]
                for r, c in extra:
                    floors[f,   r, c] = LIBRE
                    floors[f+1, r, c] = LIBRE
                    cands.append((r, c))
            chosen = [cands[i] for i in self.rng.choice(
                len(cands), size=min(n_st, len(cands)), replace=False)]
            for r, c in chosen:
                floors[f,   r, c] = ESCALERA_SUBIR
                floors[f+1, r, c] = ESCALERA_BAJAR
                stairs_up.append((f, r, c))
                stairs_down.append((f+1, r, c))
        return stairs_up, stairs_down

    def _ensure_connectivity(self, floors, stairs_up, stairs_down):
        H, W, F = self.cfg.grid_h, self.cfg.grid_w, self.cfg.n_floors
        su = set(stairs_up)
        sd = set(stairs_down)

        def neighbors(f, r, c):
            nbs = []
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < H and 0 <= nc < W and floors[f, nr, nc] != PARED:
                    nbs.append((f, nr, nc))
            if (f, r, c) in su   and f+1 < F: nbs.append((f+1, r, c))
            if (f, r, c) in sd   and f   > 0: nbs.append((f-1, r, c))
            return nbs

        def bfs(start):
            vis = {start}; q = deque([start])
            while q:
                node = q.popleft()
                for nb in neighbors(*node):
                    if nb not in vis:
                        vis.add(nb); q.append(nb)
            return vis

        all_free = [(f,r,c) for f in range(F) for r in range(H) for c in range(W)
                    if floors[f,r,c] != PARED]

        visited = set(); best = set()
        for cell in all_free:
            if cell not in visited:
                comp = bfs(cell)
                visited |= comp
                if len(comp) > len(best):
                    best = comp

        # Expansión iterativa
        for _ in range(500):
            if len(best) >= self.cfg.min_connected:
                break
            frontier = set()
            for f,r,c in best:
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 < nr < H-1 and 0 < nc < W-1 and floors[f,nr,nc] == PARED:
                        frontier.add((f,nr,nc))
            if not frontier: break
            batch = list(frontier)
            self.rng.shuffle(batch)
            for f,r,c in batch[:50]:
                floors[f,r,c] = LIBRE
                best.add((f,r,c))

        # Aislar celdas desconectadas
        for f,r,c in all_free:
            if (f,r,c) not in best:
                floors[f,r,c] = PARED
                su.discard((f,r,c)); sd.discard((f,r,c))

        return floors, [s for s in su if s in best], [s for s in sd if s in best]


# ============================================================
# Entorno MARL de Exploración 3D
# ============================================================

class MARLExploration3D(Env):
    """
    Entorno MARL cooperativo: N agentes exploran juntos un mapa 3D.

    API Gymnasium
    -------------
    obs, info                          = env.reset(seed=42)
    obs, rewards, terminated, truncated, info = env.step(actions)

    actions : dict {agent_id (int) : action (int)}
    obs     : dict {agent_id (int) : np.ndarray shape (obs_dim,)}
    rewards : dict {agent_id (int) : float}

    Estado global (para critic centralizado)
    ----------------------------------------
    state = env.get_global_state()  →  np.ndarray shape (global_state_dim,)

    Métricas del episodio
    ---------------------
    info["coverage_ratio"]       fracción explorada [0,1]
    info["steps_to_50"]          paso en que se alcanzó el 50% (-1 si no)
    info["steps_to_75"]          ídem 75%
    info["steps_to_100"]         ídem 100%
    info["redundancy_ratio"]     pasos redundantes / total
    info["team_spread"]          distancia media entre pares de agentes
    """

    def __init__(self, config: Optional[ExplorationConfig] = None):
        self.cfg = config or ExplorationConfig()
        self._base_rng = np.random.default_rng(self.cfg.seed)

        self.n_agents  = self.cfg.n_agents
        self.H         = self.cfg.grid_h
        self.W         = self.cfg.grid_w
        self.F         = self.cfg.n_floors
        self.n_actions = N_ACTIONS

        # Dimensión de observación (calculada una vez)
        r = self.cfg.obs_radius
        view_cells = (2*r+1)**2
        self.obs_dim = (
            view_cells       # tipo de celda local
            + view_cells     # mapa de visita local
            + 5              # [floor_n, r_n, c_n, global_cov, step_n]
            + self.n_agents * 5  # otros agentes [df,dr,dc,cov_norm,in_range]
            + (self.cfg.msg_dim if self.cfg.comm_mode == "explicit" else 0)
        )

        self.observation_space = Box(0.0, 1.0, shape=(self.obs_dim,))
        self.action_space      = Discrete(N_ACTIONS)

        # Estado del entorno (se inicializa en reset)
        self.floors:       np.ndarray = None   # (F, H, W) int32
        self.visited:      np.ndarray = None   # (F, H, W) bool
        self.stairs_up:    List[Tuple] = []
        self.stairs_down:  List[Tuple] = []
        self._su_set:      set = set()
        self._sd_set:      set = set()
        self._free_cells:  List[Tuple] = []
        self._n_free:      int = 0

        # Arrays vectorizados de agentes
        self.agent_f   = np.zeros(self.n_agents, dtype=np.int32)
        self.agent_r   = np.zeros(self.n_agents, dtype=np.int32)
        self.agent_c   = np.zeros(self.n_agents, dtype=np.int32)
        self.agent_alive = np.ones(self.n_agents, dtype=bool)

        # Mensajes
        self.agent_msgs = np.zeros((self.n_agents, self.cfg.msg_dim), dtype=np.float32)

        # Contadores
        self.step_count          = 0
        self._cells_visited      = 0
        self._redundant_steps    = 0
        self._total_steps        = 0
        self._steps_to_50        = -1
        self._steps_to_75        = -1
        self._steps_to_100       = -1
        self._agent_new_cells    = np.zeros(self.n_agents, dtype=np.int32)

    # ----------------------------------------------------------
    # Propiedades útiles
    # ----------------------------------------------------------

    @property
    def coverage_ratio(self) -> float:
        if self._n_free == 0: return 0.0
        return self._cells_visited / self._n_free

    @property
    def global_state_dim(self) -> int:
        return self.n_agents * 4 + 3   # [f,r,c,alive]×N + [cov, step, n_free_norm]

    # ----------------------------------------------------------
    # Interfaz pública
    # ----------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[int, np.ndarray], Dict]:

        rng = np.random.default_rng(seed) if seed is not None else self._base_rng

        # Generar mapa nuevo
        gen = _MapGenerator(self.cfg, rng)
        self.floors, self.stairs_up, self.stairs_down = gen.generate()
        self._su_set = set(self.stairs_up)
        self._sd_set = set(self.stairs_down)

        # Mapa de visitas
        self.visited = np.zeros((self.F, self.H, self.W), dtype=bool)

        # Inventariar celdas libres
        self._free_cells = [
            (f, r, c)
            for f in range(self.F)
            for r in range(self.H)
            for c in range(self.W)
            if self.floors[f, r, c] != PARED
        ]
        self._n_free = len(self._free_cells)

        # Resetear contadores
        self.step_count       = 0
        self._cells_visited   = 0
        self._redundant_steps = 0
        self._total_steps     = 0
        self._steps_to_50     = -1
        self._steps_to_75     = -1
        self._steps_to_100    = -1
        self._agent_new_cells = np.zeros(self.n_agents, dtype=np.int32)
        self.agent_msgs[:] = 0.0

        # Colocar agentes en celdas libres distintas (espaciadas)
        self._place_agents(rng)

        # Marcar posiciones iniciales como visitadas
        for i in range(self.n_agents):
            f, r, c = int(self.agent_f[i]), int(self.agent_r[i]), int(self.agent_c[i])
            if not self.visited[f, r, c]:
                self.visited[f, r, c] = True
                self._cells_visited += 1
                self._agent_new_cells[i] += 1

        return self._get_all_obs(), self._get_info()

    def step(
        self,
        actions: Dict[int, int],
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, float], bool, bool, Dict]:

        self.step_count += 1
        self._total_steps += self.n_agents
        n   = self.n_agents
        rew = np.zeros(n, dtype=np.float64)

        act = np.array([actions.get(i, int(Action.ARRIBA)) for i in range(n)], dtype=np.int32)

        # 1. Comunicación
        if self.cfg.comm_mode == "explicit":
            self._process_comm(act)

        # 2. Calcular nuevas posiciones (movimientos en el plano)
        new_f = self.agent_f.copy()
        new_r = self.agent_r.copy()
        new_c = self.agent_c.copy()

        for i in range(n):
            a = int(act[i])
            f, r, c = int(self.agent_f[i]), int(self.agent_r[i]), int(self.agent_c[i])

            if a in (0, 1, 2, 3):   # movimiento en plano
                nr = r + int(_DR[a])
                nc = c + int(_DC[a])
                if (0 <= nr < self.H and 0 <= nc < self.W and
                        self.floors[f, nr, nc] != PARED):
                    new_r[i], new_c[i] = nr, nc
                else:
                    rew[i] += self.cfg.reward_wall

            elif a == int(Action.ESC_SUBIR):
                if (f, r, c) in self._su_set and f + 1 < self.F:
                    new_f[i] = f + 1
                else:
                    rew[i] += self.cfg.reward_wall * 0.5

            elif a == int(Action.ESC_BAJAR):
                if (f, r, c) in self._sd_set and f > 0:
                    new_f[i] = f - 1
                else:
                    rew[i] += self.cfg.reward_wall * 0.5

        # 3. Resolver colisiones entre agentes
        new_f, new_r, new_c = self._resolve_collisions(new_f, new_r, new_c)

        # 4. Aplicar movimientos
        self.agent_f = new_f
        self.agent_r = new_r
        self.agent_c = new_c

        # 5. Recompensas de exploración
        prev_visited = self._cells_visited
        for i in range(n):
            f, r, c = int(new_f[i]), int(new_r[i]), int(new_c[i])
            if not self.visited[f, r, c]:
                self.visited[f, r, c] = True
                self._cells_visited += 1
                self._agent_new_cells[i] += 1
                rew[i] += self.cfg.reward_new_cell
            else:
                rew[i] += self.cfg.reward_redundant
                self._redundant_steps += 1

        # 6. Penalización por paso
        rew += self.cfg.reward_step

        # 7. Hitos de cobertura
        cov = self.coverage_ratio
        self._check_milestones()

        # 8. Bonus de completado
        if cov >= self.cfg.coverage_target:
            bonus = self.cfg.reward_completion / n
            rew += bonus

        # 9. Recompensa de equipo
        new_cells_this_step = self._cells_visited - prev_visited
        team_rew = new_cells_this_step * self.cfg.reward_new_cell / n

        # 10. Esquema de recompensa
        if self.cfg.reward_mode == "shared":
            rew[:] = team_rew + self.cfg.reward_step
        elif self.cfg.reward_mode == "mixed":
            a = self.cfg.reward_alpha
            rew = a * rew + (1 - a) * (team_rew + self.cfg.reward_step)

        # 11. Terminación
        terminated = cov >= self.cfg.coverage_target
        truncated  = self.step_count >= self.cfg.max_steps

        return (
            self._get_all_obs(),
            {i: float(rew[i]) for i in range(n)},
            terminated,
            truncated,
            self._get_info(),
        )

    def render(self, floor: int = 0, view_size: int = 35) -> str:
        """
        Muestra una ventana alrededor del agente 0 con el mapa de visitas
        superpuesto.

        Leyenda:
          # pared   · libre no visitado   ░ visitado
          ^ esc.subir   v esc.bajar
          0-N agentes
        """
        f0 = int(self.agent_f[0])
        r0c = int(self.agent_r[0])
        c0c = int(self.agent_c[0])
        display_floor = floor if floor != f0 else f0
        half = view_size // 2
        r0, r1 = max(0, r0c - half), min(self.H, r0c + half)
        c0, c1 = max(0, c0c - half), min(self.W, c0c + half)

        agent_pos = {
            (int(self.agent_f[i]), int(self.agent_r[i]), int(self.agent_c[i])): i
            for i in range(self.n_agents)
        }

        CELL_CH = {LIBRE: "·", PARED: "#", ESCALERA_SUBIR: "^", ESCALERA_BAJAR: "v"}

        lines = [
            "─" * 60,
            f" Paso {self.step_count}/{self.cfg.max_steps}  │  "
            f"Cobertura: {self.coverage_ratio*100:.1f}%  │  "
            f"Modo: {self.cfg.comm_mode}/{self.cfg.reward_mode}",
            f" Planta mostrada: {display_floor}  │  "
            f"Agente 0: P{f0}({r0c},{c0c})",
            "─" * 60,
        ]

        for ri in range(r0, r1):
            row = ""
            for ci in range(c0, c1):
                pos = (display_floor, ri, ci)
                if pos in agent_pos:
                    row += str(agent_pos[pos])
                else:
                    cell = int(self.floors[display_floor, ri, ci])
                    if cell == PARED:
                        row += "#"
                    elif self.visited[display_floor, ri, ci]:
                        row += "░"
                    else:
                        row += CELL_CH.get(cell, "?")
            lines.append(row)

        lines.append("─" * 60)
        lines.append(f" Celdas libres: {self._n_free:,}  │  "
                     f"Visitadas: {self._cells_visited:,}  │  "
                     f"Escaleras↑: {len(self.stairs_up)}  ↓: {len(self.stairs_down)}")
        lines.append(f" Agentes: " +
                     "  ".join(
                         f"[{i}]P{int(self.agent_f[i])}({int(self.agent_r[i])},{int(self.agent_c[i])})"
                         for i in range(self.n_agents)
                     ))
        lines.append(f" Celdas nuevas/agente: {self._agent_new_cells.tolist()}")
        return "\n".join(lines)

    def close(self): pass

    # ----------------------------------------------------------
    # Estado global (CTDE)
    # ----------------------------------------------------------

    def get_global_state(self) -> np.ndarray:
        """
        Vector compacto para el critic centralizado.
        No serializa el mapa completo: usa solo posiciones + estadísticas.

          [f_i/F, r_i/H, c_i/W, alive_i] × n_agents
          [coverage_ratio, step_norm, n_free_norm]
        """
        agent_feats = np.stack([
            self.agent_f / max(1, self.F - 1),
            self.agent_r / self.H,
            self.agent_c / self.W,
            self.agent_alive.astype(np.float32),
        ], axis=1).flatten().astype(np.float32)

        global_feats = np.array([
            self.coverage_ratio,
            self.step_count / self.cfg.max_steps,
            self._n_free / max(1, self.H * self.W * self.F),
        ], dtype=np.float32)

        return np.concatenate([agent_feats, global_feats])

    # ----------------------------------------------------------
    # Observaciones
    # ----------------------------------------------------------

    def _get_all_obs(self) -> Dict[int, np.ndarray]:
        return {i: self._build_obs(i) for i in range(self.n_agents)}

    def _build_obs(self, i: int) -> np.ndarray:
        f = int(self.agent_f[i])
        r = int(self.agent_r[i])
        c = int(self.agent_c[i])
        rad = self.cfg.obs_radius
        size = 2 * rad + 1

        # --- Vista local del mapa (tipo de celda) ---
        r0, r1 = r - rad, r + rad + 1
        c0, c1 = c - rad, c + rad + 1
        gr0, gr1 = max(0, r0), min(self.H, r1)
        gc0, gc1 = max(0, c0), min(self.W, c1)

        cell_patch  = np.full((size, size), PARED,  dtype=np.int32)
        visit_patch = np.zeros((size, size),         dtype=np.float32)

        pr0 = gr0 - r0; pc0 = gc0 - c0
        cell_patch [pr0:pr0+(gr1-gr0), pc0:pc0+(gc1-gc0)] = self.floors[f, gr0:gr1, gc0:gc1]
        visit_patch[pr0:pr0+(gr1-gr0), pc0:pc0+(gc1-gc0)] = self.visited[f, gr0:gr1, gc0:gc1].astype(np.float32)

        # Marcar otros agentes en la misma planta como celda especial (valor 0.9)
        for j in range(self.n_agents):
            if j == i or int(self.agent_f[j]) != f:
                continue
            dr = int(self.agent_r[j]) - r
            dc = int(self.agent_c[j]) - c
            if abs(dr) <= rad and abs(dc) <= rad:
                cell_patch[dr + rad, dc + rad] = -1   # marcador de agente

        # Convertir a float
        cell_obs  = np.where(cell_patch == -1, 0.9,
                             _CELL_OBS[np.clip(cell_patch, 0, N_CELL_TYPES-1)]).flatten().astype(np.float32)
        visit_obs = visit_patch.flatten()

        # --- Estado propio ---
        self_obs = np.array([
            f / max(1, self.F - 1),
            r / self.H,
            c / self.W,
            self.coverage_ratio,
            self.step_count / self.cfg.max_steps,
        ], dtype=np.float32)

        # --- Otros agentes (en rango de comunicación) ---
        other_obs = np.zeros(self.n_agents * 5, dtype=np.float32)
        for j in range(self.n_agents):
            if j == i: continue
            df = int(self.agent_f[j]) - f
            dr = int(self.agent_r[j]) - r
            dc = int(self.agent_c[j]) - c
            dist = abs(df) * self.H + abs(dr) + abs(dc)
            in_range = (
                self.cfg.comm_mode in ("partial", "explicit")
                and dist <= self.cfg.comm_range
            ) or (abs(dr) <= rad and abs(dc) <= rad and df == 0)

            if in_range:
                k = j * 5
                other_obs[k]   = df / self.F
                other_obs[k+1] = dr / self.H
                other_obs[k+2] = dc / self.W
                other_obs[k+3] = self._agent_new_cells[j] / max(1, self._n_free)
                other_obs[k+4] = 1.0  # en rango

        # --- Mensaje recibido ---
        msg_obs = (self.agent_msgs[i].copy()
                   if self.cfg.comm_mode == "explicit"
                   else np.zeros(0, dtype=np.float32))

        return np.concatenate([cell_obs, visit_obs, self_obs, other_obs, msg_obs])

    # ----------------------------------------------------------
    # Comunicación explícita
    # ----------------------------------------------------------

    def _process_comm(self, act: np.ndarray):
        senders = np.where(act == int(Action.SEND_MSG))[0]
        for s in senders:
            msg = self._encode_msg(s)
            df  = np.abs(self.agent_f - self.agent_f[s])
            dr  = np.abs(self.agent_r - self.agent_r[s])
            dc  = np.abs(self.agent_c - self.agent_c[s])
            dist = df * self.H + dr + dc
            in_range = (dist <= self.cfg.comm_range) & self.agent_alive
            in_range[s] = False
            self.agent_msgs[in_range] = msg

    def _encode_msg(self, i: int) -> np.ndarray:
        msg = np.zeros(self.cfg.msg_dim, dtype=np.float32)
        d = self.cfg.msg_dim
        if d >= 1: msg[0] = self.agent_f[i] / max(1, self.F - 1)
        if d >= 2: msg[1] = self.agent_r[i] / self.H
        if d >= 3: msg[2] = self.agent_c[i] / self.W
        if d >= 4: msg[3] = self.coverage_ratio
        if d >= 5: msg[4] = self._agent_new_cells[i] / max(1, self._n_free)
        return msg

    # ----------------------------------------------------------
    # Mecánicas auxiliares
    # ----------------------------------------------------------

    def _place_agents(self, rng: np.random.Generator):
        """
        Coloca agentes intentando maximizar la separación inicial.
        Usa k-means++ simplificado: cada agente nuevo se coloca en
        la celda libre más lejana del agente ya colocado más cercano.
        """
        if not self._free_cells:
            raise RuntimeError("No hay celdas libres.")

        free = np.array(self._free_cells, dtype=np.float32)  # (N, 3)
        chosen_idx = [int(rng.integers(0, len(free)))]

        for _ in range(self.n_agents - 1):
            chosen = free[chosen_idx]          # (k, 3)
            # Distancia de cada celda libre al agente más cercano
            diffs = free[:, None, :] - chosen[None, :, :]   # (N, k, 3)
            dists = np.abs(diffs).sum(axis=2).min(axis=1)    # (N,)
            # Elegir la más lejana (con algo de aleatoriedad)
            probs = dists / (dists.sum() + 1e-8)
            chosen_idx.append(int(rng.choice(len(free), p=probs)))

        for i, idx in enumerate(chosen_idx):
            f, r, c = self._free_cells[idx]
            self.agent_f[i] = f
            self.agent_r[i] = r
            self.agent_c[i] = c
            self.agent_alive[i] = True

    def _resolve_collisions(self, new_f, new_r, new_c):
        dest   = list(zip(new_f.tolist(), new_r.tolist(), new_c.tolist()))
        counts = Counter(dest)
        for i, d in enumerate(dest):
            if counts[d] > 1:
                new_f[i] = self.agent_f[i]
                new_r[i] = self.agent_r[i]
                new_c[i] = self.agent_c[i]
        return new_f, new_r, new_c

    def _check_milestones(self):
        cov = self.coverage_ratio
        if self._steps_to_50  < 0 and cov >= 0.50: self._steps_to_50  = self.step_count
        if self._steps_to_75  < 0 and cov >= 0.75: self._steps_to_75  = self.step_count
        if self._steps_to_100 < 0 and cov >= 1.00: self._steps_to_100 = self.step_count

    # ----------------------------------------------------------
    # Info
    # ----------------------------------------------------------

    def _get_info(self) -> Dict:
        spread = self._team_spread()
        redundancy = (self._redundant_steps / max(1, self._total_steps))
        return {
            "step":              self.step_count,
            "coverage_ratio":    self.coverage_ratio,
            "cells_visited":     self._cells_visited,
            "cells_total":       self._n_free,
            "steps_to_50":       self._steps_to_50,
            "steps_to_75":       self._steps_to_75,
            "steps_to_100":      self._steps_to_100,
            "redundancy_ratio":  redundancy,
            "team_spread":       spread,
            "agent_new_cells":   self._agent_new_cells.tolist(),
            "agent_positions":   list(zip(
                self.agent_f.tolist(), self.agent_r.tolist(), self.agent_c.tolist()
            )),
        }

    def _team_spread(self) -> float:
        """Distancia Manhattan media entre todos los pares de agentes."""
        n = self.n_agents
        if n < 2: return 0.0
        total = 0.0; pairs = 0
        for i in range(n):
            for j in range(i+1, n):
                total += (
                    abs(int(self.agent_f[i]) - int(self.agent_f[j])) * self.H
                    + abs(int(self.agent_r[i]) - int(self.agent_r[j]))
                    + abs(int(self.agent_c[i]) - int(self.agent_c[j]))
                )
                pairs += 1
        return total / pairs

    def sample_action(self) -> int:
        return int(np.random.randint(0, N_ACTIONS))


# ============================================================
# Factory
# ============================================================

def make_exploration_env(
    n_agents:     int   = 6,
    comm_mode:    str   = "none",
    reward_mode:  str   = "mixed",
    reward_alpha: float = 0.4,
    obs_radius:   int   = 5,
    grid_h:       int   = 250,
    grid_w:       int   = 250,
    n_floors:     int   = 3,
    wall_density: float = 0.35,
    n_stairs:     int   = 12,
    min_connected: int  = 1500,
    max_steps:    int   = 5000,
    seed:         Optional[int] = None,
    **kwargs,
) -> MARLExploration3D:
    """
    Factory para crear el entorno de exploración con parámetros rápidos.

    Ejemplos
    --------
    # Entorno estándar sin comunicación
    env = make_exploration_env(n_agents=6)

    # Con comunicación explícita y recompensa compartida
    env = make_exploration_env(comm_mode="explicit", reward_mode="shared")

    # Grid más pequeño para pruebas rápidas
    env = make_exploration_env(grid_h=30, grid_w=30, n_floors=2, min_connected=200)

    # Reproducible con semilla fija
    env = make_exploration_env(seed=42)
    """
    cfg = ExplorationConfig(
        grid_h=grid_h, grid_w=grid_w, n_floors=n_floors,
        wall_density=wall_density, n_stairs=n_stairs,
        min_connected=min_connected,
        n_agents=n_agents, obs_radius=obs_radius,
        comm_mode=comm_mode, reward_mode=reward_mode,
        reward_alpha=reward_alpha,
        max_steps=max_steps, seed=seed,
        **kwargs,
    )
    return MARLExploration3D(cfg)


# ============================================================
# Demo + benchmark
# ============================================================

def demo(n_steps: int = 20, render: bool = True):
    print("=" * 65)
    print("MARL Exploración 3D — Demo con acciones aleatorias")
    print("=" * 65)

    env = make_exploration_env(
        n_agents=6, grid_h=250, grid_w=250, n_floors=3,
        comm_mode="explicit", reward_mode="mixed",
        reward_alpha=0.4, seed=7,
    )

    t0 = time.perf_counter()
    obs, info = env.reset(seed=7)
    t_reset = time.perf_counter() - t0

    print(f"\nEntorno         : {env.cfg.grid_h}×{env.cfg.grid_w}×{env.cfg.n_floors}")
    print(f"Agentes         : {env.n_agents}")
    print(f"obs_dim         : {env.obs_dim}")
    print(f"global_state_dim: {env.global_state_dim}")
    print(f"Celdas libres   : {info['cells_total']:,}")
    print(f"Reset time      : {t_reset:.2f}s\n")

    if render:
        print(env.render())
        print()

    total_rew = {i: 0.0 for i in range(env.n_agents)}
    t_steps = 0.0

    for step in range(n_steps):
        actions = {i: env.sample_action() for i in range(env.n_agents)}
        t0 = time.perf_counter()
        obs, rewards, term, trunc, info = env.step(actions)
        t_steps += time.perf_counter() - t0

        for i in range(env.n_agents):
            total_rew[i] += rewards[i]

        print(f"Step {step+1:03d} | "
              f"cov={info['coverage_ratio']*100:.2f}% | "
              f"visited={info['cells_visited']:,} | "
              f"spread={info['team_spread']:.1f} | "
              f"redundancy={info['redundancy_ratio']*100:.1f}%")

        if term or trunc:
            print(f"\n  Episodio terminado (term={term}, trunc={trunc})")
            break

    avg_ms = t_steps / max(1, n_steps) * 1000
    print(f"\nRecompensas acumuladas : {[round(total_rew[i],3) for i in range(env.n_agents)]}")
    print(f"Tiempo medio/step      : {avg_ms:.2f} ms  ({1000/avg_ms:.0f} steps/s)")

    if render:
        print()
        print(env.render())

    return env


def benchmark(n_episodes: int = 2, n_steps: int = 200):
    print("\nBenchmark MARLExploration3D 250×250×3")
    print("-" * 45)
    env = make_exploration_env(n_agents=6, seed=1)
    total = 0
    t0 = time.perf_counter()
    for ep in range(n_episodes):
        env.reset()
        for _ in range(n_steps):
            actions = {i: env.sample_action() for i in range(env.n_agents)}
            _, _, term, trunc, _ = env.step(actions)
            total += 1
            if term or trunc: break
    elapsed = time.perf_counter() - t0
    print(f"  {total} steps en {elapsed:.2f}s")
    print(f"  {total/elapsed:.0f} steps/s  |  "
          f"{total*env.n_agents/elapsed:.0f} agent-steps/s")


if __name__ == "__main__":
    demo(n_steps=20, render=True)
    benchmark()
