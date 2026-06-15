"""
video.py  —  Generación de vídeos MP4 para todos los equipos / fases
=====================================================================

Recorre todos los TeamConfig definidos en agente_info.py, busca el
checkpoint más avanzado disponible en checkpoints_sb3/ y genera un
vídeo MP4 por cada equipo encontrado.

El frame renderiza TODOS los pisos del mapa de MARLExploration3D
apilados verticalmente, con los agentes dibujados en su piso real.

USO
---
# Generar vídeos de todos los equipos (checkpoint más alto disponible)
    python video.py

# Sólo el equipo 5, fase 3
    python video.py --team 5 --phase 3

# Especificar directorio de checkpoints y salida
    python video.py --ckpt_dir checkpoints_sb3 --out_dir videos

ESTRUCTURA DE CHECKPOINT ESPERADA
----------------------------------
    checkpoints_sb3/model_team{TEAM_ID}_phase{PHASE_ID}.zip
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, ".")
from agente_info import ARCHETYPES, CURRICULUM, TEAMS, TeamConfig
from entrenar import SB3MultiAgentVecEnv, _build_env
from marl_exploration_3d import MARLExploration3D

# ──────────────────────────────────────────────────────────────
# PARÁMETROS VISUALES
# ──────────────────────────────────────────────────────────────

CELL_SIZE   = 6          # píxeles por celda
FPS         = 20         # fotogramas por segundo
FLOOR_GAP   = 12         # píxeles de separación entre pisos
PANEL_H     = 120        # altura del panel de info (parte superior)
MAX_STEPS   = 20_000     # límite de seguridad de frames por vídeo

# Colores BGR
COLOR_WALL      = (30,  30,  30)
COLOR_FREE      = (210, 210, 210)
COLOR_VISITED   = (40,  160,  40)
COLOR_STAIR_UP  = (200, 180,  50)
COLOR_STAIR_DN  = (50,  130, 200)
COLOR_BG        = (15,  15,  15)   # fondo del panel de info
COLOR_TEXT      = (230, 230, 230)
COLOR_TITLE     = (255, 220,  80)
COLOR_PROGRESS  = (60,  200,  60)
COLOR_PROGRESS_BG = (60, 60,  60)

# Un color distinto por agente (hasta 10)
AGENT_COLORS: List[Tuple[int, int, int]] = [
    (0,   50,  255),   # azul
    (0,  200,   50),   # verde
    (220,  0,    0),   # rojo
    (200, 160,   0),   # amarillo
    (180,   0,  220),  # violeta
    (0,  200,  200),   # cian
    (255, 100,   0),   # naranja
    (0,  180,  255),   # celeste
    (255,   0,  160),  # rosa
    (120, 255,   0),   # lima
]

# Tipos de celda según marl_exploration_3d.py
LIBRE          = 0
PARED          = 1
ESCALERA_SUBIR = 2
ESCALERA_BAJAR = 3


# ──────────────────────────────────────────────────────────────
# BÚSQUEDA DE CHECKPOINTS
# ──────────────────────────────────────────────────────────────

def _find_all_checkpoints(
    ckpt_dir: Path,
    team_id:  int,
    phase_id: Optional[int] = None,
) -> List[Tuple[int, Path]]:
    """
    Devuelve la lista de (phase_idx, path) de todos los checkpoints
    disponibles para el equipo, ordenados por fase ascendente.

    Si phase_id no es None, devuelve solo ese checkpoint concreto
    (lista vacía si no existe).
    """
    found: List[Tuple[int, Path]] = []

    phases = (
        [CURRICULUM[phase_id]]
        if phase_id is not None
        else CURRICULUM
    )

    for phase in phases:
        p = ckpt_dir / f"model_team{team_id}_phase{phase.idx}.zip"
        if p.exists():
            found.append((phase.idx, p))

    return found


# ──────────────────────────────────────────────────────────────
# RENDERIZADO DE FRAME
# ──────────────────────────────────────────────────────────────

def _render_map(env: MARLExploration3D, cell: int) -> np.ndarray:
    """
    Devuelve una imagen BGR con todos los pisos apilados verticalmente.
    Cada celda ocupa `cell`×`cell` píxeles.
    Escaleras, celdas visitadas y agentes se pintan encima.
    """
    F, H, W = env.F, env.H, env.W
    img_h = F * H * cell + (F - 1) * FLOOR_GAP
    img_w = W * cell
    img   = np.full((img_h, img_w, 3), COLOR_FREE, dtype=np.uint8)

    for f in range(F):
        y_off = f * (H * cell + FLOOR_GAP)
        grid    = env.floors[f]       # (H, W) int32
        visited = env.visited[f]      # (H, W) bool

        # ── Pintar celdas ──────────────────────────────────────
        for r in range(H):
            for c in range(W):
                y0 = y_off + r * cell
                y1 = y0 + cell
                x0 = c * cell
                x1 = x0 + cell
                cell_type = int(grid[r, c])

                if cell_type == PARED:
                    color = COLOR_WALL
                elif visited[r, c]:
                    color = COLOR_VISITED
                elif cell_type == ESCALERA_SUBIR:
                    color = COLOR_STAIR_UP
                elif cell_type == ESCALERA_BAJAR:
                    color = COLOR_STAIR_DN
                else:
                    color = COLOR_FREE

                img[y0:y1, x0:x1] = color

        # ── Etiqueta de piso ──────────────────────────────────
        cv2.putText(
            img,
            f"P{f}",
            (4, y_off + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (80, 80, 80),
            1,
            cv2.LINE_AA,
        )

    # ── Dibujar agentes ───────────────────────────────────────
    for i in range(env.n_agents):
        f  = int(env.agent_f[i])
        r  = int(env.agent_r[i])
        c  = int(env.agent_c[i])
        y_off = f * (H * cell + FLOOR_GAP)
        y0 = y_off + r * cell
        y1 = y0 + cell
        x0 = c * cell
        x1 = x0 + cell
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        # Círculo centrado en la celda
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        radius = max(1, cell // 2 - 1)
        cv2.circle(img, (cx, cy), radius, color, -1)

    return img


def _render_panel(
    env:        MARLExploration3D,
    team:       TeamConfig,
    phase_name: str,
    step:       int,
    panel_w:    int,
) -> np.ndarray:
    """Panel de información superior (fondo oscuro)."""
    panel = np.full((PANEL_H, panel_w, 3), COLOR_BG, dtype=np.uint8)

    cov     = env.coverage_ratio * 100
    visited = env._cells_visited
    total   = env._n_free

    # ── Título ────────────────────────────────────────────────
    title = f"Equipo {team.id}: {team.name}  {team.label}  |  Fase: {phase_name}"
    cv2.putText(panel, title, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TITLE, 1, cv2.LINE_AA)

    # ── Composición de arquetipos ──────────────────────────────
    arch_str = "  ".join(
        f"[A{aid}:{ARCHETYPES[aid].name}]" for aid in team.unique_archetypes()
    )
    cv2.putText(panel, arch_str, (10, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)

    # ── Métricas ──────────────────────────────────────────────
    metrics = (
        f"Paso: {step:>6d}   "
        f"Cobertura: {cov:5.1f}%   "
        f"Visitadas: {visited:,} / {total:,}   "
        f"Agentes: {env.n_agents}"
    )
    cv2.putText(panel, metrics, (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)

    # ── Leyenda de agentes ────────────────────────────────────
    x_leg = 10
    for i in range(env.n_agents):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        cv2.circle(panel, (x_leg + 8, 90), 7, color, -1)
        aid   = team.composition[i]
        label = f"A{i}:{ARCHETYPES[aid].name[:4]}"
        cv2.putText(panel, label, (x_leg + 18, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
        x_leg += 90

    # ── Barra de progreso ─────────────────────────────────────
    bar_x, bar_y, bar_w, bar_h = 10, 106, panel_w - 20, 8
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  COLOR_PROGRESS_BG, -1)
    filled = int(bar_w * env.coverage_ratio)
    if filled > 0:
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h),
                      COLOR_PROGRESS, -1)

    return panel


# ──────────────────────────────────────────────────────────────
# GENERACIÓN DE VÍDEO PARA UN EQUIPO
# ──────────────────────────────────────────────────────────────

def generate_video(
    team_id:    int,
    phase_id:   int,
    ckpt_path:  Path,
    out_path:   Path,
    cell:       int = CELL_SIZE,
    fps:        int = FPS,
) -> None:
    """Genera el vídeo MP4 para un equipo+fase con el modelo PPO guardado."""

    phase = CURRICULUM[phase_id]
    team  = TEAMS[team_id]

    print(f"\n{'─'*60}")
    print(f"  Equipo {team_id}: {team.name}  |  Fase {phase_id}: {phase.name}")
    print(f"  Mapa: {phase.grid_h}×{phase.grid_w}×{phase.n_floors}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Salida    : {out_path}")
    print(f"{'─'*60}")

    # Construir entorno igual que en entrenar.py
    base_env, dominant_id = _build_env(phase=phase, team_id=team_id)
    vec_env = SB3MultiAgentVecEnv(base_env, dominant_id)

    # Cargar modelo
    model = PPO.load(str(ckpt_path), env=vec_env)

    # Primera observación
    obs = vec_env.reset()

    # Calcular dimensiones del frame
    F, H, W  = base_env.F, base_env.H, base_env.W
    map_h    = F * H * cell + (F - 1) * FLOOR_GAP
    map_w    = W * cell
    frame_h  = PANEL_H + map_h
    frame_w  = map_w

    # Writer de vídeo
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_w, frame_h),
    )

    if not writer.isOpened():
        print(f"  [ERROR] No se pudo abrir el VideoWriter para {out_path}")
        vec_env.close()
        return

    step    = 0
    done    = False

    while not done and step < MAX_STEPS:
        # ── Renderizar frame ──────────────────────────────────
        map_img   = _render_map(base_env, cell)
        panel_img = _render_panel(base_env, team, phase.name, step, frame_w)

        frame = np.vstack([panel_img, map_img])
        writer.write(frame)

        # ── Paso del modelo ───────────────────────────────────
        actions, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = vec_env.step(actions)
        done = bool(np.any(dones))
        step += 1

        if step % 500 == 0:
            print(
                f"    step={step:>6d}  "
                f"cov={base_env.coverage_ratio*100:.1f}%  "
                f"visited={base_env._cells_visited:,}/{base_env._n_free:,}"
            )

    # Frame final
    map_img   = _render_map(base_env, cell)
    panel_img = _render_panel(base_env, team, phase.name, step, frame_w)
    frame = np.vstack([panel_img, map_img])
    writer.write(frame)

    writer.release()
    vec_env.close()

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
            "Genera vídeos MP4 a partir de checkpoints PPO. "
            "Por defecto genera un vídeo por CADA checkpoint disponible "
            "(todos los equipos × todas las fases entrenadas)."
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

    # Descubrir todos los checkpoints a procesar
    work: List[Tuple[int, int, Path]] = []   # (team_id, phase_id, ckpt_path)
    for tid in team_ids:
        found = _find_all_checkpoints(ckpt_dir, tid, args.phase)
        for phase_id, ckpt_path in found:
            work.append((tid, phase_id, ckpt_path))

    print(f"\n{'='*60}")
    print(f"  GENERADOR DE VÍDEOS — MARL Exploración 3D")
    print(f"  Checkpoints : {ckpt_dir.resolve()}")
    print(f"  Salida      : {out_dir.resolve()}")
    print(f"  Vídeos a generar: {len(work)}")
    for tid, pid, cp in work:
        print(f"    equipo {tid} ({TEAMS[tid].name}) — fase {pid}  →  {cp.name}")
    if not work:
        print("  [!] No se encontró ningún checkpoint. Entrena primero con entrenar.py")
    print(f"{'='*60}\n")

    generated: List[Path] = []
    errors:    List[str]  = []

    for i, (tid, phase_id, ckpt_path) in enumerate(work, 1):
        out_path = out_dir / f"team{tid}_{TEAMS[tid].name}_phase{phase_id}.mp4"
        print(f"[{i}/{len(work)}] Equipo {tid} — Fase {phase_id}")
        try:
            generate_video(
                team_id   = tid,
                phase_id  = phase_id,
                ckpt_path = ckpt_path,
                out_path  = out_path,
                cell      = args.cell,
                fps       = args.fps,
            )
            generated.append(out_path)
        except Exception as exc:
            msg = f"Equipo {tid} / Fase {phase_id}: {exc}"
            print(f"\n  [ERROR] {msg}")
            import traceback; traceback.print_exc()
            errors.append(msg)

    # Resumen final
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