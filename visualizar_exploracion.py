"""
visualizar_exploracion.py — Visualizador del entorno MARL Exploración 3D
=========================================================================
Muestra en tiempo real el progreso de exploración cooperativa de N agentes.

Paneles
-------
  Izquierda  : mapa de exploración de cada planta (3 subplots)
               · libre no visitado  ░ visitado  # pared  ^ v escaleras
               Cada agente tiene su propio color y trayectoria
  Derecha    : métricas en tiempo real
               - Curva de cobertura acumulada
               - Contribución por agente (barras)
               - Estadísticas del episodio

Modos
-----
  --steps N       pasos aleatorios automáticos (demo)
  --interactive   control con teclado (WASD + teclas especiales)
  --compare       genera imagen comparando 4 semillas
  --save          guarda PNG del mapa completo
  --no-display    modo sin ventana (solo guarda archivos)

Controles interactivos
----------------------
  W/↑ S/↓ A/← D/→   mover agente 0
  U subir escalera   J bajar escalera
  1-6                seleccionar agente activo
  R reset            Q salir
"""

from __future__ import annotations

import argparse
import sys
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from collections import deque
from typing import List, Dict, Optional

sys.path.insert(0, ".")
from marl_exploration_3d import (
    make_exploration_env, MARLExploration3D,
    LIBRE, PARED, ESCALERA_SUBIR, ESCALERA_BAJAR,
    Action,
)


# ============================================================
# Paleta y estilos
# ============================================================

BG_DARK   = "#0D1117"
BG_PANEL  = "#161B22"
BG_CELL   = "#1F2937"
COL_TEXT  = "#E6EDF3"
COL_DIM   = "#8B949E"
COL_GRID  = "#30363D"

# Colores de agentes (hasta 8)
AGENT_COLORS = [
    "#FF6B6B", "#4ECDC4", "#FFE66D", "#A8E6CF",
    "#FF8B94", "#B4A7E5", "#FFEAA7", "#81ECEC",
]

# Colormap para celdas visitadas vs no visitadas
_VISIT_CMAP = LinearSegmentedColormap.from_list(
    "exploration",
    ["#1F2937",   # no visitado (libre)
     "#264653",   # visitado reciente
     "#2A9D8F"],  # visitado hace tiempo
)

# Mapa base de tipos de celda
_BASE_COLORS = {
    LIBRE:          "#1F2937",   # no visitado
    PARED:          "#0D1117",   # pared
    ESCALERA_SUBIR: "#E76F51",   # escalera subir
    ESCALERA_BAJAR: "#457B9D",   # escalera bajar
}

_VISITED_COLOR = "#2A9D8F"


# ============================================================
# Visualizador
# ============================================================

class ExplorationVisualizer:
    """
    Visualizador en tiempo real del entorno MARLExploration3D.
    """

    KEY_MAP = {
        "w": Action.ARRIBA,    "up":    Action.ARRIBA,
        "s": Action.ABAJO,     "down":  Action.ABAJO,
        "a": Action.IZQUIERDA, "left":  Action.IZQUIERDA,
        "d": Action.DERECHA,   "right": Action.DERECHA,
        "u": Action.ESC_SUBIR,
        "j": Action.ESC_BAJAR,
    }

    def __init__(self, env: MARLExploration3D, seed: Optional[int] = None):
        self.env   = env
        self.seed  = seed
        self.obs   = None
        self.info  = None
        self.done  = False
        self.total_reward = {i: 0.0 for i in range(env.n_agents)}
        self.active_agent = 0   # agente controlado en modo interactivo

        # Historial para curva de cobertura
        self._cov_history:  List[float] = []
        self._step_history: List[int]   = []

        # Trayectorias por agente (últimos K pasos)
        K = 80
        self._traj = [deque(maxlen=K) for _ in range(env.n_agents)]

        # Figura
        self._build_figure()

    # ----------------------------------------------------------
    # Construcción de la figura
    # ----------------------------------------------------------

    def _build_figure(self):
        n   = self.env.n_agents
        F   = self.env.F

        self.fig = plt.figure(figsize=(22, 10), facecolor=BG_DARK)
        self.fig.suptitle(
            f"MARL Exploración Cooperativa 3D — "
            f"{self.env.H}×{self.env.W}×{F} plantas  │  {n} agentes",
            color=COL_TEXT, fontsize=13, fontweight="bold", y=0.99,
        )

        # Layout: F plantas | curva cobertura | barras contribución | stats
        gs = gridspec.GridSpec(
            2, F + 2,
            figure=self.fig,
            width_ratios=[1]*F + [0.9, 0.7],
            height_ratios=[2.2, 1],
            wspace=0.12, hspace=0.35,
            left=0.04, right=0.97, top=0.94, bottom=0.06,
        )

        # Subplots de plantas (fila 0, columnas 0..F-1)
        self.ax_floors = []
        for f in range(F):
            ax = self.fig.add_subplot(gs[0, f])
            ax.set_facecolor(BG_DARK)
            ax.set_title(f"Planta {f}", color=COL_TEXT, fontsize=9, pad=3)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_edgecolor(COL_GRID)
            self.ax_floors.append(ax)

        # Curva de cobertura (fila 0, col F)
        self.ax_cov = self.fig.add_subplot(gs[0, F])
        self.ax_cov.set_facecolor(BG_PANEL)
        self.ax_cov.set_title("Cobertura (%)", color=COL_TEXT, fontsize=9, pad=3)
        self.ax_cov.set_xlabel("Paso", color=COL_DIM, fontsize=7)
        self.ax_cov.tick_params(colors=COL_DIM, labelsize=7)
        for sp in self.ax_cov.spines.values(): sp.set_edgecolor(COL_GRID)

        # Barras de contribución por agente (fila 0, col F+1)
        self.ax_contrib = self.fig.add_subplot(gs[0, F+1])
        self.ax_contrib.set_facecolor(BG_PANEL)
        self.ax_contrib.set_title("Celdas nuevas / agente", color=COL_TEXT, fontsize=9, pad=3)
        self.ax_contrib.tick_params(colors=COL_DIM, labelsize=7)
        for sp in self.ax_contrib.spines.values(): sp.set_edgecolor(COL_GRID)

        # Panel de stats (fila 1, cols 0..F-1)
        self.ax_stats = self.fig.add_subplot(gs[1, :F])
        self.ax_stats.set_facecolor(BG_PANEL)
        self.ax_stats.set_xticks([]); self.ax_stats.set_yticks([])
        for sp in self.ax_stats.spines.values(): sp.set_edgecolor(COL_GRID)

        # Barra de progreso global (fila 1, cols F..F+1)
        self.ax_progress = self.fig.add_subplot(gs[1, F:])
        self.ax_progress.set_facecolor(BG_PANEL)
        self.ax_progress.set_xticks([]); self.ax_progress.set_yticks([])
        for sp in self.ax_progress.spines.values(): sp.set_edgecolor(COL_GRID)

        # Leyenda de agentes
        legend_patches = [
            mpatches.Patch(color=AGENT_COLORS[i % len(AGENT_COLORS)], label=f"Agente {i}")
            for i in range(self.env.n_agents)
        ] + [
            mpatches.Patch(color=_VISITED_COLOR, label="Visitado"),
            mpatches.Patch(color=_BASE_COLORS[LIBRE], label="Libre"),
            mpatches.Patch(color=_BASE_COLORS[PARED], label="Pared"),
            mpatches.Patch(color=_BASE_COLORS[ESCALERA_SUBIR], label="Esc. ↑"),
            mpatches.Patch(color=_BASE_COLORS[ESCALERA_BAJAR], label="Esc. ↓"),
        ]
        self.fig.legend(
            handles=legend_patches,
            loc="lower center", ncol=min(len(legend_patches), 10),
            fontsize=7, facecolor=BG_PANEL, labelcolor=COL_TEXT,
            framealpha=0.9, bbox_to_anchor=(0.5, -0.01),
        )

        # Imágenes de los mapas (inicializadas en reset)
        self._floor_imgs  = [None] * F
        self._agent_dots  = [[] for _ in range(F)]   # scatter por planta
        self._traj_lines  = [[] for _ in range(F)]   # líneas de trayectoria
        self._cov_line    = None
        self._bar_container = None

    # ----------------------------------------------------------
    # Reset y dibujo inicial
    # ----------------------------------------------------------

    def reset(self):
        env = self.env
        self.obs, self.info = env.reset(seed=self.seed)
        self.done = False
        self.total_reward = {i: 0.0 for i in range(env.n_agents)}
        self._cov_history  = [0.0]
        self._step_history = [0]
        for t in self._traj: t.clear()
        for i in range(env.n_agents):
            self._traj[i].append((
                int(env.agent_f[i]), int(env.agent_r[i]), int(env.agent_c[i])
            ))
        self._full_draw()

    def _full_draw(self):
        """Dibuja todo desde cero (llamado en reset)."""
        env = self.env
        F   = env.F

        for f in range(F):
            ax = self.ax_floors[f]
            ax.clear()
            ax.set_facecolor(BG_DARK)
            ax.set_title(f"Planta {f}", color=COL_TEXT, fontsize=9, pad=3)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_edgecolor(COL_GRID)

            # Imagen base del mapa
            rgb = self._make_floor_rgb(f)
            self._floor_imgs[f] = ax.imshow(
                rgb, interpolation="nearest", aspect="equal", origin="upper"
            )

            # Trayectorias (líneas vacías al inicio)
            self._traj_lines[f] = []
            for i in range(env.n_agents):
                line, = ax.plot([], [], "-",
                                color=AGENT_COLORS[i % len(AGENT_COLORS)],
                                alpha=0.4, linewidth=0.8, zorder=3)
                self._traj_lines[f].append(line)

            # Puntos de agentes
            self._agent_dots[f] = []
            for i in range(env.n_agents):
                sc = ax.scatter([], [], s=30,
                                color=AGENT_COLORS[i % len(AGENT_COLORS)],
                                zorder=5, edgecolors="white", linewidths=0.5)
                self._agent_dots[f].append(sc)

        # Curva de cobertura
        self.ax_cov.clear()
        self.ax_cov.set_facecolor(BG_PANEL)
        self.ax_cov.set_title("Cobertura (%)", color=COL_TEXT, fontsize=9, pad=3)
        self.ax_cov.set_xlabel("Paso", color=COL_DIM, fontsize=7)
        self.ax_cov.set_ylim(0, 105)
        self.ax_cov.tick_params(colors=COL_DIM, labelsize=7)
        self.ax_cov.axhline(y=100, color=COL_DIM, linestyle="--",
                            alpha=0.4, linewidth=0.8)
        for sp in self.ax_cov.spines.values(): sp.set_edgecolor(COL_GRID)
        self._cov_line, = self.ax_cov.plot(
            [0], [0], color="#4ECDC4", linewidth=1.5
        )

        # Actualizar posiciones iniciales
        self._update_agents()
        self._update_contrib()
        self._update_stats()
        self._update_progress()
        plt.pause(0.05)

    # ----------------------------------------------------------
    # Actualización incremental
    # ----------------------------------------------------------

    def _update_draw(self):
        env = self.env

        # 1. Actualizar imágenes de mapa (zonas nuevas visitadas)
        for f in range(env.F):
            rgb = self._make_floor_rgb(f)
            self._floor_imgs[f].set_data(rgb)

        # 2. Posiciones y trayectorias de agentes
        self._update_agents()

        # 3. Curva de cobertura
        self._cov_history.append(self.info["coverage_ratio"] * 100)
        self._step_history.append(env.step_count)
        self._cov_line.set_data(self._step_history, self._cov_history)
        self.ax_cov.set_xlim(0, max(10, env.step_count + 1))
        self.ax_cov.relim(); self.ax_cov.autoscale_view(scaley=False)

        # 4. Contribución por agente
        self._update_contrib()

        # 5. Estadísticas
        self._update_stats()

        # 6. Barra de progreso
        self._update_progress()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _update_agents(self):
        env = self.env
        for i in range(env.n_agents):
            f  = int(env.agent_f[i])
            r  = int(env.agent_r[i])
            c  = int(env.agent_c[i])
            self._traj[i].append((f, r, c))

            for fl in range(env.F):
                # Punto del agente
                if f == fl:
                    self._agent_dots[fl][i].set_offsets([[c, r]])
                    self._agent_dots[fl][i].set_alpha(1.0)
                else:
                    self._agent_dots[fl][i].set_offsets(np.empty((0, 2)))

                # Trayectoria en la planta correcta
                pts = [(rc, cc) for (ff, rc, cc) in self._traj[i] if ff == fl]
                if len(pts) >= 2:
                    rs, cs = zip(*pts)
                    self._traj_lines[fl][i].set_data(cs, rs)
                else:
                    self._traj_lines[fl][i].set_data([], [])

    def _update_contrib(self):
        ax = self.ax_contrib
        ax.clear()
        ax.set_facecolor(BG_PANEL)
        ax.set_title("Celdas nuevas / agente", color=COL_TEXT, fontsize=9, pad=3)
        ax.tick_params(colors=COL_DIM, labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor(COL_GRID)

        n   = self.env.n_agents
        vals = self.env._agent_new_cells.tolist()
        bars = ax.barh(
            range(n), vals,
            color=[AGENT_COLORS[i % len(AGENT_COLORS)] for i in range(n)],
            edgecolor=BG_DARK, linewidth=0.5,
        )
        ax.set_yticks(range(n))
        ax.set_yticklabels([f"A{i}" for i in range(n)],
                           color=COL_DIM, fontsize=7)
        ax.set_xlabel("Celdas", color=COL_DIM, fontsize=7)
        mx = max(vals) if vals else 1
        ax.set_xlim(0, mx * 1.15)
        for bar, val in zip(bars, vals):
            ax.text(val + mx*0.02, bar.get_y() + bar.get_height()/2,
                    str(val), va="center", color=COL_TEXT, fontsize=6)

    def _update_stats(self):
        ax = self.ax_stats
        ax.clear()
        ax.set_facecolor(BG_PANEL)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor(COL_GRID)

        env  = self.env
        info = self.info or {}
        cov  = info.get("coverage_ratio", 0.0) * 100
        vis  = info.get("cells_visited", 0)
        tot  = info.get("cells_total", 1)
        red  = info.get("redundancy_ratio", 0.0) * 100
        sprd = info.get("team_spread", 0.0)
        s50  = info.get("steps_to_50",  -1)
        s75  = info.get("steps_to_75",  -1)
        s100 = info.get("steps_to_100", -1)

        def fmt_milestone(v):
            return str(v) if v >= 0 else "—"

        cols = [
            ("Paso",           f"{env.step_count}/{env.cfg.max_steps}"),
            ("Cobertura",      f"{cov:.2f}%"),
            ("Visitadas",      f"{vis:,} / {tot:,}"),
            ("Redundancia",    f"{red:.1f}%"),
            ("Dispersión",     f"{sprd:.1f}"),
            ("Hito 50%",       fmt_milestone(s50)),
            ("Hito 75%",       fmt_milestone(s75)),
            ("Hito 100%",      fmt_milestone(s100)),
            ("Comm",           env.cfg.comm_mode),
            ("Reward",         f"{env.cfg.reward_mode} α={env.cfg.reward_alpha:.1f}"),
        ]

        x_step = 1.0 / len(cols)
        for k, (label, val) in enumerate(cols):
            x = x_step * k + x_step * 0.5
            ax.text(x, 0.72, label, transform=ax.transAxes,
                    color=COL_DIM, fontsize=7, ha="center", va="center")
            ax.text(x, 0.30, val,   transform=ax.transAxes,
                    color=COL_TEXT, fontsize=9, ha="center", va="center",
                    fontweight="bold")

    def _update_progress(self):
        ax = self.ax_progress
        ax.clear()
        ax.set_facecolor(BG_PANEL)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor(COL_GRID)

        cov = self.info.get("coverage_ratio", 0.0) if self.info else 0.0
        ax.set_title("Progreso de exploración", color=COL_TEXT, fontsize=9, pad=3)

        # Barra de progreso
        ax.barh([0], [cov], color="#2A9D8F", edgecolor=BG_DARK, height=0.4)
        ax.barh([0], [1.0], color=BG_CELL,  edgecolor=COL_GRID,  height=0.4, zorder=0)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(-0.5, 0.5)
        ax.text(max(cov - 0.05, 0.02), 0, f"{cov*100:.1f}%",
                va="center", color="white", fontsize=11, fontweight="bold")

        # Hitos
        for pct, label in [(0.5, "50%"), (0.75, "75%"), (1.0, "100%")]:
            ax.axvline(pct, color=COL_DIM, linestyle="--",
                       alpha=0.5, linewidth=0.8)
            ax.text(pct, 0.48, label, ha="center", color=COL_DIM, fontsize=6,
                    transform=ax.get_xaxis_transform())

    # ----------------------------------------------------------
    # Construcción del RGB del mapa
    # ----------------------------------------------------------

    def _make_floor_rgb(self, f: int) -> np.ndarray:
        """
        Genera imagen RGB del mapa de la planta f mostrando:
        - negro:    paredes
        - gris oscuro: libre no visitado
        - verde:    visitado
        - naranja:  escalera subir
        - azul:     escalera bajar
        """
        H, W   = self.env.H, self.env.W
        floors = self.env.floors[f]
        vis    = self.env.visited[f]

        # Base de colores
        rgb = np.zeros((H, W, 3), dtype=np.float32)

        # Celdas libres no visitadas
        m = (floors == LIBRE) & (~vis)
        rgb[m] = [0.12, 0.16, 0.22]

        # Celdas visitadas
        m = (floors == LIBRE) & vis
        rgb[m] = [0.16, 0.61, 0.56]

        # Paredes
        m = floors == PARED
        rgb[m] = [0.05, 0.07, 0.09]

        # Escaleras subir
        m = floors == ESCALERA_SUBIR
        rgb[m] = [0.91, 0.44, 0.32]

        # Escaleras bajar
        m = floors == ESCALERA_BAJAR
        rgb[m] = [0.27, 0.48, 0.62]

        return rgb

    # ----------------------------------------------------------
    # Modos de ejecución
    # ----------------------------------------------------------

    def run_random(self, n_steps: int = 200, delay: float = 0.02):
        print(f"\nEjecutando {n_steps} pasos aleatorios...")
        for step in range(n_steps):
            if self.done:
                cov = self.info.get("coverage_ratio", 0) * 100
                print(f"  Episodio terminado (cov={cov:.1f}%). Reseteando...")
                self.reset()

            actions = {i: self.env.sample_action() for i in range(self.env.n_agents)}
            obs, rew, term, trunc, info = self.env.step(actions)
            for i in range(self.env.n_agents):
                self.total_reward[i] += rew[i]
            self.info = info
            self.done = term or trunc
            self._update_draw()

            if delay > 0:
                time.sleep(delay)

        cov = self.info.get("coverage_ratio", 0) * 100 if self.info else 0
        print(f"Cobertura final: {cov:.2f}%")
        plt.show()

    def run_interactive(self):
        print("\nModo interactivo:")
        print("  W/↑ S/↓ A/← D/→  U=SubirEsc  J=BajarEsc")
        print("  1-6 seleccionar agente activo  R=Reset  Q=Salir\n")

        def on_key(event):
            key = event.key.lower() if event.key else ""

            if key == "q":
                plt.close("all"); return
            if key == "r":
                self.reset(); return
            if key in "123456789":
                idx = int(key) - 1
                if idx < self.env.n_agents:
                    self.active_agent = idx
                    print(f"  Agente activo: {idx}")
                return
            if self.done:
                self.reset(); return
            if key not in self.KEY_MAP:
                return

            # Mover agente activo; el resto se queda quieto
            action = {i: int(Action.ARRIBA) for i in range(self.env.n_agents)}
            action[self.active_agent] = int(self.KEY_MAP[key])

            obs, rew, term, trunc, info = self.env.step(action)
            for i in range(self.env.n_agents):
                self.total_reward[i] += rew[i]
            self.info = info
            self.done = term or trunc
            self._update_draw()

            cov = info.get("coverage_ratio", 0) * 100
            status = " *** COMPLETADO ***" if term else (" (truncado)" if trunc else "")
            print(f"Step {self.env.step_count:4d} | "
                  f"cov={cov:.2f}% | "
                  f"rew[{self.active_agent}]={rew[self.active_agent]:+.3f}{status}")

        self.fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show()


# ============================================================
# Comparación multi-semilla
# ============================================================

def comparar_semillas(seeds: List[int], n_steps: int = 300):
    """
    Ejecuta n_steps pasos aleatorios en entornos con distintas semillas
    y compara la curva de cobertura resultante.
    """
    print(f"\nComparando {len(seeds)} semillas durante {n_steps} pasos...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG_DARK)
    fig.suptitle("Comparación de exploración por semilla (acciones aleatorias)",
                 color=COL_TEXT, fontsize=11, y=1.01)

    ax_cov   = axes[0]
    ax_final = axes[1]
    ax_cov.set_facecolor(BG_PANEL)
    ax_final.set_facecolor(BG_PANEL)

    for ax in axes:
        ax.tick_params(colors=COL_DIM, labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor(COL_GRID)

    final_covs = []
    colors_s   = ["#FF6B6B","#4ECDC4","#FFE66D","#A8E6CF"]

    for s_idx, seed in enumerate(seeds):
        color = colors_s[s_idx % len(colors_s)]
        print(f"  Semilla {seed}...", end="", flush=True)

        env = make_exploration_env(
            n_agents=6, grid_h=250, grid_w=250, n_floors=3,
            comm_mode="none", reward_mode="mixed",
            min_connected=1500, max_steps=n_steps, seed=seed,
        )
        env.reset(seed=seed)
        cov_hist = [0.0]
        step_hist = [0]

        for _ in range(n_steps):
            actions = {i: env.sample_action() for i in range(env.n_agents)}
            _, _, term, trunc, info = env.step(actions)
            cov_hist.append(info["coverage_ratio"] * 100)
            step_hist.append(env.step_count)
            if term or trunc: break

        final_cov = cov_hist[-1]
        final_covs.append(final_cov)
        ax_cov.plot(step_hist, cov_hist, color=color, linewidth=1.5,
                    label=f"Semilla {seed} → {final_cov:.1f}%")
        print(f" {final_cov:.1f}%")

    ax_cov.set_title("Curva de cobertura", color=COL_TEXT, fontsize=10, pad=4)
    ax_cov.set_xlabel("Paso", color=COL_DIM, fontsize=8)
    ax_cov.set_ylabel("Cobertura (%)", color=COL_DIM, fontsize=8)
    ax_cov.axhline(100, color=COL_DIM, linestyle="--", alpha=0.3)
    ax_cov.legend(fontsize=8, facecolor=BG_PANEL, labelcolor=COL_TEXT)
    ax_cov.set_ylim(0, 105)

    ax_final.bar(
        [f"Semilla {s}" for s in seeds], final_covs,
        color=colors_s[:len(seeds)], edgecolor=BG_DARK,
    )
    ax_final.set_title(f"Cobertura final tras {n_steps} pasos",
                       color=COL_TEXT, fontsize=10, pad=4)
    ax_final.set_ylabel("Cobertura (%)", color=COL_DIM, fontsize=8)
    ax_final.set_ylim(0, 110)
    for i, v in enumerate(final_covs):
        ax_final.text(i, v + 1, f"{v:.1f}%", ha="center",
                      color=COL_TEXT, fontsize=9, fontweight="bold")

    plt.tight_layout()
    out = "/mnt/user-data/outputs/comparacion_exploracion.png"
    plt.savefig(out, dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\nGuardado: {out}")
    plt.show()


# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualizador MARL Exploración 3D"
    )
    parser.add_argument("--seed",        type=int,   default=None)
    parser.add_argument("--n-agents",    type=int,   default=6)
    parser.add_argument("--comm",        type=str,   default="none",
                        choices=["none","partial","explicit"])
    parser.add_argument("--reward",      type=str,   default="mixed",
                        choices=["individual","shared","mixed"])
    parser.add_argument("--alpha",       type=float, default=0.4)
    parser.add_argument("--steps",       type=int,   default=300)
    parser.add_argument("--delay",       type=float, default=0.02)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--compare",     action="store_true")
    parser.add_argument("--no-display",  action="store_true")
    args = parser.parse_args()

    if args.no_display:
        matplotlib.use("Agg")

    if args.compare:
        base = args.seed or 0
        comparar_semillas([base, base+1, base+2, base+3],
                          n_steps=args.steps)
        return

    print(f"\nCreando entorno MARL Exploración 3D...")
    print(f"  Agentes: {args.n_agents} | Comm: {args.comm} | "
          f"Reward: {args.reward}(α={args.alpha}) | Semilla: {args.seed}")

    env = make_exploration_env(
        n_agents     = args.n_agents,
        comm_mode    = args.comm,
        reward_mode  = args.reward,
        reward_alpha = args.alpha,
        grid_h=250, grid_w=250, n_floors=3,
        min_connected=1500,
        max_steps    = args.steps,
        seed         = args.seed,
    )

    t0 = time.perf_counter()
    obs, info = env.reset(seed=args.seed)
    print(f"  Reset: {time.perf_counter()-t0:.2f}s | "
          f"Celdas libres: {info['cells_total']:,} | "
          f"obs_dim: {env.obs_dim}")

    viz = ExplorationVisualizer(env, seed=args.seed)
    viz.reset()

    if args.interactive:
        viz.run_interactive()
    else:
        viz.run_random(n_steps=args.steps, delay=args.delay)


if __name__ == "__main__":
    main()
