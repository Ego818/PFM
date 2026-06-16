from __future__ import annotations
"""
video.py  —  Generación de vídeos MP4 para todos los equipos / fases
=====================================================================

Recorre todos los TeamConfig definidos en agente_info.py, busca los
checkpoints disponibles en checkpoints_sb3/ y genera un vídeo MP4 por
cada equipo/fase encontrado.

ARQUITECTURA MULTI-POLÍTICA (compatible con entrenar2.py / entrenar.py v3)
---------------------------------------------------------------------------
Cada arquetipo único del equipo tiene su propio modelo PPO:
    checkpoints_sb3/model_team{T}_phase{P}_arch{A}.zip

En cada paso de inferencia se consulta el modelo del arquetipo
correspondiente a cada agente y se combinan las acciones antes de
llamar a env.step().

Si sólo existe un fichero sin sufijo _arch (legado de versiones anteriores):
    checkpoints_sb3/model_team{T}_phase{P}.zip
se carga como modelo único compartido por todos los agentes (modo legado).

USO
---
# Generar vídeos de todos los equipos (checkpoint más alto disponible)
    python video.py

# Sólo el equipo 5, fase 3
    python video.py --team 5 --phase 3

# Especificar directorio de checkpoints y salida
    python video.py --ckpt_dir checkpoints_sb3 --out_dir videos
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, ".")
from agente_info import ARCHETYPES, CURRICULUM, TEAMS, TeamConfig
from entrenar2 import _build_env, ArchetypeVecEnv
from marl_exploration_3d import MARLExploration3D
from visualizar_exploracion import ExplorationVisualizer


# ──────────────────────────────────────────────────────────────
# PARÁMETROS VISUALES
# ──────────────────────────────────────────────────────────────

CELL_SIZE         = 6
FPS               = 20
FLOOR_GAP         = 12
PANEL_H           = 120
MAX_STEPS         = 20_000

COLOR_WALL        = (30,  30,  30)
COLOR_FREE        = (210, 210, 210)
COLOR_VISITED     = (40,  160,  40)
COLOR_STAIR_UP    = (200, 180,  50)
COLOR_STAIR_DN    = (50,  130, 200)
COLOR_BG          = (15,  15,  15)
COLOR_TEXT        = (230, 230, 230)
COLOR_TITLE       = (255, 220,  80)
COLOR_PROGRESS    = (60,  200,  60)
COLOR_PROGRESS_BG = (60,  60,  60)

AGENT_COLORS: List[Tuple[int, int, int]] = [
    (0,   50,  255),
    (0,  200,   50),
    (220,   0,   0),
    (200, 160,   0),
    (180,   0,  220),
    (0,  200,  200),
    (255, 100,   0),
    (0,  180,  255),
    (255,   0,  160),
    (120, 255,   0),
]

LIBRE          = 0
PARED          = 1
ESCALERA_SUBIR = 2
ESCALERA_BAJAR = 3


# ──────────────────────────────────────────────────────────────
# BÚSQUEDA DE CHECKPOINTS  (multi-política)
# ──────────────────────────────────────────────────────────────

def _find_arch_checkpoints(
    ckpt_dir: Path,
    team_id:  int,
    phase_id: int,
) -> Optional[Dict[int, Path]]:
    """
    Busca los checkpoints por arquetipo para un equipo/fase.

    Devuelve un dict {arch_id: path} si se encuentran TODOS los
    arquetipos únicos del equipo, o None si falta alguno.

    Modo legado: si no existen archivos _arch pero sí existe el .zip
    sin sufijo, devuelve {-1: path} como señal de modo legado.
    """
    team         = TEAMS[team_id]
    unique_archs = team.unique_archetypes()

    result: Dict[int, Path] = {}
    for arch_id in unique_archs:
        p = ckpt_dir / f"model_team{team_id}_phase{phase_id}_arch{arch_id}.zip"
        if p.exists():
            result[arch_id] = p

    if len(result) == len(unique_archs):
        return result  # todos los arquetipos presentes

    # Fallback: checkpoint legado sin sufijo _arch
    legacy = ckpt_dir / f"model_team{team_id}_phase{phase_id}.zip"
    if legacy.exists():
        return {-1: legacy}

    return None


def _find_all_checkpoints(
    ckpt_dir: Path,
    team_id:  int,
    phase_id: Optional[int] = None,
) -> List[Tuple[int, Dict[int, Path]]]:
    """
    Devuelve lista de (phase_idx, arch_paths_dict) para el equipo dado.
    Si phase_id no es None, filtra solo esa fase.
    """
    phases = [CURRICULUM[phase_id]] if phase_id is not None else CURRICULUM
    found: List[Tuple[int, Dict[int, Path]]] = []
    for phase in phases:
        arch_paths = _find_arch_checkpoints(ckpt_dir, team_id, phase.idx)
        if arch_paths is not None:
            found.append((phase.idx, arch_paths))
    return found


# ──────────────────────────────────────────────────────────────
# INFERENCIA MULTI-POLÍTICA
# ──────────────────────────────────────────────────────────────

def _predict_all(
    models:        Dict[int, PPO],
    vec_envs:      Dict[int, ArchetypeVecEnv],
    arch_to_agents: Dict[int, List[int]],
    n_agents:      int,
) -> np.ndarray:
    """
    Recolecta acciones de todos los modelos (uno por arquetipo) y las
    combina en un único array de longitud n_agents.
    """
    actions = np.zeros(n_agents, dtype=np.int64)
    for arch_id, model in models.items():
        vec_env      = vec_envs[arch_id]
        agent_indices = arch_to_agents[arch_id]
        obs          = vec_env.get_obs()           # (n_arch_agents, obs_dim)
        acts, _      = model.predict(obs, deterministic=True)
        for local_i, global_i in enumerate(agent_indices):
            actions[global_i] = int(acts[local_i])
    return actions


# ──────────────────────────────────────────────────────────────
# GENERACIÓN DE VÍDEO PARA UN EQUIPO/FASE
# ──────────────────────────────────────────────────────────────

def generate_video(
    team_id:    int,
    phase_id:   int,
    arch_paths: Dict[int, Path],
    out_path:   Path,
    cell:       int = CELL_SIZE,
    fps:        int = FPS,
) -> None:
    """
    Genera el vídeo MP4 para un equipo+fase.

    arch_paths: dict {arch_id: ckpt_path} (multi-política)
                o    {-1: ckpt_path}      (modo legado, un solo modelo)
    """
    phase = CURRICULUM[phase_id]
    team  = TEAMS[team_id]

    print(f"\n{'─'*60}")
    print(f"  Equipo {team_id}: {team.name}  |  Fase {phase_id}: {phase.name}")
    print(f"  Mapa: {phase.grid_h}×{phase.grid_w}×{phase.n_floors}")
    for aid, p in arch_paths.items():
        label = "legado" if aid == -1 else f"arquetipo {aid} ({ARCHETYPES[aid].name})"
        print(f"  Checkpoint [{label}]: {p}")
    print(f"  Salida: {out_path}")
    print(f"{'─'*60}")

    # ── Construir entorno base ──────────────────────────────────
    base_env, dominant_arch_id = _build_env(phase=phase, team_id=team_id)

    # ── Decidir modo: multi-política o legado ───────────────────
    legacy_mode = (-1 in arch_paths)

    if legacy_mode:
        # Modo legado: un único VecEnv con todos los agentes
        vec_env_legacy = ArchetypeVecEnv(
            env           = base_env,
            archetype_id  = dominant_arch_id,
            agent_indices = list(range(team.n_agents)),
            obs_dict_init = dict(enumerate(
                base_env.reset()[0].values()
                if hasattr(base_env.reset()[0], "values")
                else {i: v for i, v in enumerate(base_env.reset()[0])}
            )),
        )
        model_legacy = PPO.load(str(arch_paths[-1]), env=vec_env_legacy)
        obs_legacy   = vec_env_legacy.reset()
        models      = None
        vec_envs    = None
        arch_to_agents = None
    else:
        # Multi-política: un VecEnv + PPO por arquetipo
        obs_dict_init, _ = base_env.reset()

        arch_to_agents: Dict[int, List[int]] = {}
        for agent_idx, arch_id in enumerate(team.composition):
            arch_to_agents.setdefault(arch_id, []).append(agent_idx)

        vec_envs: Dict[int, ArchetypeVecEnv] = {}
        models:   Dict[int, PPO]             = {}

        for arch_id, agent_indices in arch_to_agents.items():
            vec_env = ArchetypeVecEnv(
                env           = base_env,
                archetype_id  = arch_id,
                agent_indices = agent_indices,
                obs_dict_init = obs_dict_init,
            )
            model = PPO.load(str(arch_paths[arch_id]), env=vec_env)
            vec_envs[arch_id] = vec_env
            models[arch_id]   = model

    # ── Configurar visualizador ─────────────────────────────────
    viz = ExplorationVisualizer(base_env)
    viz.reset()
    viz.fig.canvas.draw()

    h = viz.fig.canvas.get_width_height()[1]
    w = viz.fig.canvas.get_width_height()[0]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    if not writer.isOpened():
        print(f"  [ERROR] No se pudo abrir el VideoWriter para {out_path}")
        base_env.close()
        return

    step = 0
    done = False

    while not done and step < MAX_STEPS:
        # ── Renderizar frame ────────────────────────────────────
        viz.info = {
            "coverage_ratio": base_env.coverage_ratio,
            "cells_visited":  base_env._cells_visited,
            "cells_total":    base_env._n_free,
        }
        viz._update_draw()
        viz.fig.canvas.draw()
        frame = np.asarray(viz.fig.canvas.buffer_rgba())[:, :, :3]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)

        # ── Inferencia y paso ───────────────────────────────────
        if legacy_mode:
            actions_arr, _ = model_legacy.predict(obs_legacy, deterministic=True)
            # step() espera dict {agent_id: action}
            actions_dict = {i: int(a) for i, a in enumerate(actions_arr)}
            obs_dict_next, rewards, terminated, truncated, info = base_env.step(actions_dict)
            done_env = terminated or truncated
            if done_env:
                obs_dict_next, _ = base_env.reset()
            # Convertir obs_dict a array para el siguiente predict
            obs_legacy = np.array(
                [obs_dict_next[i] for i in range(team.n_agents)], dtype=np.float32
            )
            done = done_env
        else:
            # Recolectar acciones de cada modelo
            actions_dict: Dict[int, int] = {}
            for arch_id, model in models.items():
                vec_env      = vec_envs[arch_id]
                agent_indices = arch_to_agents[arch_id]
                obs          = vec_env.get_obs()
                acts, _      = model.predict(obs, deterministic=True)
                for local_i, global_i in enumerate(agent_indices):
                    actions_dict[global_i] = int(acts[local_i])

            obs_dict_next, rewards, terminated, truncated, info = base_env.step(actions_dict)
            done_env = terminated or truncated
            if done_env:
                obs_dict_next, _ = base_env.reset()

            # Distribuir obs a cada ArchetypeVecEnv
            for arch_id, vec_env in vec_envs.items():
                vec_env.ingest_step(obs_dict_next, rewards, done_env, info)

            done = done_env

        step += 1

        if step % 500 == 0:
            print(
                f"    step={step:>6d}  "
                f"cov={base_env.coverage_ratio*100:.1f}%  "
                f"visited={base_env._cells_visited:,}/{base_env._n_free:,}"
            )

    # ── Frame final ─────────────────────────────────────────────
    viz.info = {
        "coverage_ratio": base_env.coverage_ratio,
        "cells_visited":  base_env._cells_visited,
        "cells_total":    base_env._n_free,
    }
    viz._update_draw()
    viz.fig.canvas.draw()
    frame = np.asarray(viz.fig.canvas.buffer_rgba())[:, :, :3]
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    writer.write(frame)

    writer.release()
    plt.close(viz.fig)
    base_env.close()

    print(
        f"\n  ✓ Vídeo guardado: {out_path}"
        f"\n    Pasos: {step}  |  Cobertura final: {base_env.coverage_ratio*100:.1f}%"
    )


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Genera vídeos MP4 a partir de checkpoints PPO (multi-política). "
            "Por defecto genera un vídeo por cada checkpoint disponible."
        )
    )
    parser.add_argument(
        "--team", type=int, default=None,
        help="Filtrar por equipo concreto (1-10). Si se omite, procesa todos.",
    )
    parser.add_argument(
        "--phase", type=int, default=None,
        help="Filtrar por fase concreta (0-5). Si se omite, procesa todas las fases disponibles.",
    )
    parser.add_argument(
        "--ckpt_dir", type=str, default="checkpoints_sb3",
        help="Directorio donde están los .zip de los modelos.",
    )
    parser.add_argument(
        "--out_dir", type=str, default="videos",
        help="Directorio donde se guardarán los MP4.",
    )
    parser.add_argument(
        "--cell", type=int, default=CELL_SIZE,
        help=f"Tamaño de celda en píxeles (default={CELL_SIZE}).",
    )
    parser.add_argument(
        "--fps", type=int, default=FPS,
        help=f"Fotogramas por segundo (default={FPS}).",
    )
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    out_dir  = Path(args.out_dir)

    team_ids: List[int] = (
        [args.team] if args.team is not None else list(TEAMS.keys())
    )

    # Descubrir todos los checkpoints
    work: List[Tuple[int, int, Dict[int, Path]]] = []
    for tid in team_ids:
        for phase_id, arch_paths in _find_all_checkpoints(ckpt_dir, tid, args.phase):
            work.append((tid, phase_id, arch_paths))

    print(f"\n{'='*60}")
    print(f"  GENERADOR DE VÍDEOS — MARL Exploración 3D (multi-política)")
    print(f"  Checkpoints : {ckpt_dir.resolve()}")
    print(f"  Salida      : {out_dir.resolve()}")
    print(f"  Vídeos a generar: {len(work)}")
    for tid, pid, ap in work:
        mode = "legado" if -1 in ap else f"{len(ap)} arquetipo(s)"
        print(f"    equipo {tid} ({TEAMS[tid].name}) — fase {pid}  [{mode}]")
    if not work:
        print("  [!] No se encontró ningún checkpoint. Entrena primero con entrenar.py")
    print(f"{'='*60}\n")

    generated: List[Path] = []
    errors:    List[str]  = []

    for i, (tid, phase_id, arch_paths) in enumerate(work, 1):
        out_path = out_dir / f"team{tid}_{TEAMS[tid].name}_phase{phase_id}.mp4"
        print(f"[{i}/{len(work)}] Equipo {tid} — Fase {phase_id}")
        try:
            generate_video(
                team_id    = tid,
                phase_id   = phase_id,
                arch_paths = arch_paths,
                out_path   = out_path,
                cell       = args.cell,
                fps        = args.fps,
            )
            generated.append(out_path)
        except Exception as exc:
            msg = f"Equipo {tid} / Fase {phase_id}: {exc}"
            print(f"\n  [ERROR] {msg}")
            import traceback; traceback.print_exc()
            errors.append(msg)

    print(f"\n{'='*60}")
    print(f"  RESUMEN")
    print(f"  Generados : {len(generated)}")
    for p in generated:
        print(f"    ✓ {p}")
    if errors:
        print(f"  Errores   : {len(errors)}")
        for e in errors:
            print(f"    ✗ {e}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()