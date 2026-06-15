"""
check_env.py — Verificación de dependencias del proyecto
=========================================================
Ejecuta este script antes de entrenar o generar vídeos para
confirmar que todas las dependencias están correctamente instaladas.

    python check_env.py
"""

import sys
import importlib

# ─────────────────────────────────────────────────────────────────
# (paquete_a_importar, nombre_legible, versión_mínima)
# ─────────────────────────────────────────────────────────────────
REQUIRED = [
    ("numpy",            "NumPy",             "1.24.0"),
    ("scipy",            "SciPy",             "1.10.0"),
    ("gymnasium",        "Gymnasium",          "0.29.0"),
    ("stable_baselines3","Stable-Baselines3",  "2.3.0"),
    ("torch",            "PyTorch",            "2.1.0"),
    ("cv2",              "OpenCV (cv2)",        "4.8.0"),
    ("matplotlib",       "Matplotlib",         "3.7.0"),
    ("tensorboard",      "TensorBoard",        "2.14.0"),
]

OPTIONAL = [
    ("rich",   "Rich",   "13.0.0"),   # salida SB3 más legible
    ("pandas", "Pandas", "2.0.0"),    # métricas SB3
]

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def _version_tuple(v: str):
    try:
        return tuple(int(x) for x in v.split(".")[:3])
    except ValueError:
        return (0, 0, 0)


def check(packages, label: str) -> int:
    """Retorna el número de fallos."""
    print(f"\n{BOLD}{label}{RESET}")
    print("─" * 55)
    failures = 0

    for mod_name, display_name, min_ver in packages:
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", "?")
            if ver != "?" and _version_tuple(ver) < _version_tuple(min_ver):
                status = f"{YELLOW}⚠  {ver} (mín: {min_ver}){RESET}"
                failures += 1
            else:
                status = f"{GREEN}✓  {ver}{RESET}"
        except ImportError:
            status = f"{RED}✗  NO INSTALADO{RESET}"
            failures += 1

        print(f"  {display_name:<25} {status}")

    return failures


def check_torch_device():
    print(f"\n{BOLD}DISPOSITIVO PyTorch{RESET}")
    print("─" * 55)
    try:
        import torch
        print(f"  CPU disponible        {GREEN}✓{RESET}")
        cuda = torch.cuda.is_available()
        if cuda:
            name = torch.cuda.get_device_name(0)
            print(f"  CUDA disponible       {GREEN}✓  {name}{RESET}")
        else:
            print(f"  CUDA disponible       {YELLOW}✗  (solo CPU){RESET}")
        mps = getattr(torch.backends, "mps", None)
        if mps and mps.is_available():
            print(f"  MPS (Apple M-series)  {GREEN}✓{RESET}")
    except ImportError:
        pass


def check_sb3_ppo():
    print(f"\n{BOLD}STABLE-BASELINES3 — PPO importable{RESET}")
    print("─" * 55)
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecEnv
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.buffers import RolloutBuffer
        print(f"  PPO, VecEnv, BaseCallback, RolloutBuffer  {GREEN}✓{RESET}")
    except ImportError as e:
        print(f"  {RED}✗  {e}{RESET}")


def check_cv2_codec():
    print(f"\n{BOLD}OPENCV — codec mp4v{RESET}")
    print("─" * 55)
    try:
        import cv2, tempfile, os
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        writer = cv2.VideoWriter(
            tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), 24, (64, 64)
        )
        ok = writer.isOpened()
        writer.release()
        os.unlink(tmp.name)
        if ok:
            print(f"  Codec mp4v  {GREEN}✓{RESET}")
        else:
            print(f"  Codec mp4v  {YELLOW}⚠  no disponible (falta ffmpeg o codec){RESET}")
    except Exception as e:
        print(f"  {RED}✗  {e}{RESET}")


def check_project_files():
    print(f"\n{BOLD}ARCHIVOS DEL PROYECTO{RESET}")
    print("─" * 55)
    files = [
        "marl_exploration_3d.py",
        "agente_info.py",
        "entrenar.py",
        "video.py",
        "train_all.py",
    ]
    for f in files:
        exists = __import__("pathlib").Path(f).exists()
        mark   = f"{GREEN}✓{RESET}" if exists else f"{RED}✗  no encontrado{RESET}"
        print(f"  {f:<30} {mark}")


def main():
    print(f"\n{'='*55}")
    print(f"  {BOLD}CHECK DE ENTORNO — MARL Exploración 3D{RESET}")
    print(f"  Python {sys.version.split()[0]}  |  {sys.executable}")
    print(f"{'='*55}")

    fails  = check(REQUIRED, "DEPENDENCIAS OBLIGATORIAS")
    fails += check(OPTIONAL, "DEPENDENCIAS OPCIONALES")

    check_torch_device()
    check_sb3_ppo()
    check_cv2_codec()
    check_project_files()

    print(f"\n{'='*55}")
    if fails == 0:
        print(f"  {GREEN}{BOLD}Todo listo. Puedes entrenar y generar vídeos.{RESET}")
    else:
        print(f"  {RED}{BOLD}{fails} problema(s) detectado(s).{RESET}")
        print(f"  Ejecuta:  pip install -r requirements.txt")
    print(f"{'='*55}\n")
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()