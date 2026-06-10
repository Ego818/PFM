"""
grabar_video.py
===============

Genera un vídeo MP4 a partir de un checkpoint PPO (.zip)
entrenado con entrenar.py.
"""

from stable_baselines3 import PPO
import numpy as np
import cv2

from entrenar import (
    _build_env,
    SB3MultiAgentVecEnv
)

# --------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------

TEAM_ID = 1
PHASE_ID = 5

MODEL_PATH = f"checkpoints_sb3/model_team{TEAM_ID}_phase{PHASE_ID}.zip"

VIDEO_PATH = f"team{TEAM_ID}_phase{PHASE_ID}.mp4"

FPS = 15
CELL_SIZE = 8

# --------------------------------------------------
# COLORES BGR
# --------------------------------------------------

COLOR_WALL = (40, 40, 40)
COLOR_FREE = (220, 220, 220)
COLOR_VISITED = (0, 180, 0)

AGENT_COLORS = [
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 128, 255),
    (128, 0, 255),
    (0, 255, 128),
]

# --------------------------------------------------
# FRAME RGB
# --------------------------------------------------

def build_frame(env):

    floor = int(env.agent_f[0])

    grid = env.floors[floor]
    visited = env.visited[floor]

    h, w = grid.shape

    img = np.zeros(
        (h * CELL_SIZE, w * CELL_SIZE, 3),
        dtype=np.uint8
    )

    img[:] = COLOR_FREE

    walls = grid == 1
    img.reshape(h, CELL_SIZE, w, CELL_SIZE, 3) \
       .swapaxes(1, 2)[walls] = COLOR_WALL

    visited_mask = visited & (grid != 1)

    img.reshape(h, CELL_SIZE, w, CELL_SIZE, 3) \
       .swapaxes(1, 2)[visited_mask] = COLOR_VISITED

    for i in range(env.n_agents):

        if env.agent_f[i] != floor:
            continue

        r = int(env.agent_r[i])
        c = int(env.agent_c[i])

        y0 = r * CELL_SIZE
        y1 = (r + 1) * CELL_SIZE

        x0 = c * CELL_SIZE
        x1 = (c + 1) * CELL_SIZE

        color = AGENT_COLORS[i % len(AGENT_COLORS)]

        cv2.rectangle(
            img,
            (x0, y0),
            (x1, y1),
            color,
            thickness=-1
        )

    cov = env.coverage_ratio * 100

    cv2.putText(
        img,
        f"Coverage: {cov:.1f}%",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2,
    )

    return img


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    base_env, arch = _build_env(
        phase=__import__("agente_info").CURRICULUM[PHASE_ID],
        team_id=TEAM_ID
    )

    vec_env = SB3MultiAgentVecEnv(base_env, arch)

    model = PPO.load(
        MODEL_PATH,
        env=vec_env
    )

    obs = vec_env.reset()

    frame = build_frame(base_env)

    h, w = frame.shape[:2]

    writer = cv2.VideoWriter(
        VIDEO_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (w, h)
    )

    done = False
    step = 0

    while not done:

        actions, _ = model.predict(
            obs,
            deterministic=True
        )

        obs, rewards, dones, infos = vec_env.step(actions)

        frame = build_frame(base_env)

        writer.write(frame)

        step += 1

        done = dones.any()

        if step % 100 == 0:
            print(
                f"step={step} "
                f"cov={base_env.coverage_ratio*100:.2f}%"
            )

    writer.release()

    print()
    print(f"Vídeo guardado en: {VIDEO_PATH}")
    print(
        f"Cobertura final: "
        f"{base_env.coverage_ratio*100:.2f}%"
    )


if __name__ == "__main__":
    main()