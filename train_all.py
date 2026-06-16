# train_all.py

import sys
import subprocess
from multiprocessing import Pool


REQUIRED_PACKAGES = [
    "numpy",
    "scipy",
    "matplotlib",
    "gymnasium",
    "torch",
    "stable-baselines3[extra]",
]


def install_requirements():
    print("Comprobando dependencias...")

    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        *REQUIRED_PACKAGES
    ])

    print("Dependencias listas.\n")


def train(team_id):
    cmd = [
        sys.executable,
        "entrenar.py",
        "--team",
        str(team_id)
    ]

    print(f"Lanzando equipo {team_id}")

    subprocess.run(cmd, check=False)

    print(f"Equipo {team_id} terminado")


if __name__ == "__main__":

    # instalar una sola vez
    install_requirements()

    # entrenar los 10 equipos en paralelo
    with Pool(processes=10) as pool:
        pool.map(train, range(6, 11))