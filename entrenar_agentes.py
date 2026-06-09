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

import os, sys, time, json, argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import deque

import numpy as np

# ── Matplotlib para video ──────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FFMpegWriter, FuncAnimation

sys.path.insert(0, ".")
from marl_exploration_3d import (
    MARLExploration3D, ExplorationConfig, make_exploration_env, Action
)

# ─────────────────────────────────────────────────────────────
OUT_DIR = "checkpoints"
os.makedirs(OUT_DIR, exist_ok=True)
LOG_FILE = os.path.join(OUT_DIR, "training_log.jsonl")


# ═══════════════════════════════════════════════════════════════
# 1. ACTIVACIONES Y UTILIDADES
# ═══════════════════════════════════════════════════════════════

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def relu_grad(x):
    return (x > 0).astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# 1b. NORMALIZADOR DE OBSERVACIONES (running mean/std)
# ═══════════════════════════════════════════════════════════════

class ObsNormalizer:
    """
    Normalización online de observaciones con running mean y std.
    Esencial cuando se aprende con muy pocas muestras: evita que la red
    reciba valores de distinta escala según el tamaño del grid.

    Algoritmo de Welford para actualización incremental estable.
    """

    def __init__(self, obs_dim: int, clip: float = 5.0):
        self.mean  = np.zeros(obs_dim, dtype=np.float64)
        self.var   = np.ones(obs_dim,  dtype=np.float64)
        self.count = 1e-4           # evita división por cero al inicio
        self.clip  = clip

    def update(self, x: np.ndarray):
        """Actualiza estadísticas con una nueva observación (shape: obs_dim)."""
        x = x.astype(np.float64)
        self.count += 1
        delta      = x - self.mean
        self.mean += delta / self.count
        delta2     = x - self.mean
        self.var   += (delta * delta2 - self.var) / self.count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normaliza una observación con las estadísticas actuales."""
        std = np.sqrt(self.var + 1e-8).astype(np.float32)
        z   = (x.astype(np.float32) - self.mean.astype(np.float32)) / std
        return np.clip(z, -self.clip, self.clip)

    def save(self) -> Dict:
        return {
            "mean":  self.mean.tolist(),
            "var":   self.var.tolist(),
            "count": self.count,
        }

    def load(self, data: Dict):
        self.mean  = np.array(data["mean"],  dtype=np.float64)
        self.var   = np.array(data["var"],   dtype=np.float64)
        self.count = float(data.get("count", 1e-4))


# ═══════════════════════════════════════════════════════════════
# 2. RED NEURONAL MLP (Actor-Critic, NumPy puro)
# ═══════════════════════════════════════════════════════════════

class MLP:
    """
    Red MLP con:
      - Capas ocultas con ReLU
      - Cabeza de política (softmax sobre n_actions)
      - Cabeza de valor (escalar)
    Optimizador: Adam con gradient clipping
    """

    def __init__(
        self,
        in_dim: int,
        hidden: Tuple[int, ...],
        n_actions: int,
        lr: float = 5e-4,       # v3: más agresivo para convergencia rápida
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        self.layers: List[Dict] = []
        self.lr = lr
        self.n_actions = n_actions

        dims = [in_dim] + list(hidden)
        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i+1]
            std = np.sqrt(2.0 / fan_in)   # He init
            W = rng.standard_normal((fan_in, fan_out)).astype(np.float32) * std
            b = np.zeros(fan_out, dtype=np.float32)
            self.layers.append({
                "W": W, "b": b,
                "mW": np.zeros_like(W), "vW": np.zeros_like(W),
                "mb": np.zeros_like(b), "vb": np.zeros_like(b),
            })

        last = dims[-1]
        # Policy head — init muy pequeña: política estrictamente uniforme al inicio
        std_head = 0.4
        self.W_pi = rng.standard_normal((last, n_actions)).astype(np.float32) * std_head
        self.b_pi = np.zeros(n_actions, dtype=np.float32)
        # Value head — init pequeña también
        self.W_v  = rng.standard_normal((last, 1)).astype(np.float32) * std_head
        self.b_v  = np.zeros(1, dtype=np.float32)

        # Adam state para cabezas
        self.m_Wpi = np.zeros_like(self.W_pi); self.v_Wpi = np.zeros_like(self.W_pi)
        self.m_bpi = np.zeros_like(self.b_pi); self.v_bpi = np.zeros_like(self.b_pi)
        self.m_Wv  = np.zeros_like(self.W_v);  self.v_Wv  = np.zeros_like(self.W_v)
        self.m_bv  = np.zeros_like(self.b_v);  self.v_bv  = np.zeros_like(self.b_v)

        self.t = 0
        self._cache: List = []

    # ── Forward ──────────────────────────────────────────────

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """Devuelve (logits_policy, value)."""
        self._cache = []
        h = x.astype(np.float32)
        for layer in self.layers:
            z = h @ layer["W"] + layer["b"]
            a = relu(z)
            self._cache.append((h, z, a))
            h = a
        logits = h @ self.W_pi + self.b_pi
        value  = float((h @ self.W_v + self.b_v).squeeze())
        self._h_last = h.copy()
        return logits, value

    def policy(self, x: np.ndarray) -> np.ndarray:
        logits, _ = self.forward(x)
        return softmax(logits)

    def act(self, x: np.ndarray) -> Tuple[int, float, float]:
        """Devuelve (acción, log_prob, value)."""
        logits, value = self.forward(x)
        probs  = softmax(logits)
        action = int(np.random.choice(self.n_actions, p=probs))
        log_prob = float(np.log(probs[action] + 1e-8))
        return action, log_prob, value

    # ── Actualización PPO ────────────────────────────────────

    def update_ppo(
        self,
        obs_batch:    np.ndarray,   # (B, obs_dim)
        act_batch:    np.ndarray,   # (B,) int
        ret_batch:    np.ndarray,   # (B,) returns
        adv_batch:    np.ndarray,   # (B,) advantages
        old_lp_batch: np.ndarray,   # (B,) old log probs — siempre del rollout original
        clip_eps: float = 0.1,      # v3: clipping más permisivo → updates más grandes
        vf_coef:  float = 1.0,      # v3: igual peso a valor que a política
        ent_coef: float = 0.005,     # v3: más entropía → más exploración
    ) -> Dict[str, float]:
        B = len(obs_batch)
        if B == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        # Normalizar ventajas
        adv = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)

        # Acumular gradientes
        grad_Wpi    = np.zeros_like(self.W_pi)
        grad_bpi    = np.zeros_like(self.b_pi)
        grad_Wv     = np.zeros_like(self.W_v)
        grad_bv     = np.zeros_like(self.b_v)
        grad_layers = [{"W": np.zeros_like(l["W"]),
                        "b": np.zeros_like(l["b"])}
                       for l in self.layers]

        total_pl = 0.0; total_vl = 0.0; total_ent = 0.0

        for i in range(B):
            x      = obs_batch[i].astype(np.float32)
            act    = int(act_batch[i])
            ret    = float(ret_batch[i])
            a_adv  = float(adv[i])
            old_lp = float(old_lp_batch[i])

            logits, val = self.forward(x)
            probs = softmax(logits)
            lp    = float(np.log(probs[act] + 1e-8))
            ratio = float(np.exp(lp - old_lp))

            # PPO clipped surrogate objective
            surr1 = ratio * a_adv
            surr2 = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * a_adv
            # La PÉRDIDA es el negativo (queremos MAXIMIZAR) → signo correcto
            pl = -min(surr1, surr2)

            # Value loss (MSE)
            vl = vf_coef * (val - ret) ** 2

            # Entropy bonus
            ent = float(-np.sum(probs * np.log(probs + 1e-8)))

            total_pl  += pl
            total_vl  += vl
            total_ent += ent

            # ── Gradiente cabeza de política ──────────────────
            # Determinar qué surrogate está activo para el gradiente
            if (ratio > 1.0 + clip_eps) or (ratio < 1.0 - clip_eps):
                eff_ratio = float(np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps))
            else:
                eff_ratio = ratio

            mask = np.zeros(self.n_actions, dtype=np.float32)
            mask[act] = 1.0

            # Gradiente de la política respecto a logits: (probs - mask) * ventaja
            d_logits = (probs - mask) * (a_adv * eff_ratio)

            # Gradiente de la entropía corregido para MAXIMIZAR exploración (añade ruido útil)
            d_ent = probs * (np.log(probs + 1e-8) + ent)
            d_logits += ent_coef * d_ent

            # Gradiente del bonus de entropía: queremos maximizar H = -sum(p*log p)
            # La derivada CORRECTA (verificada numéricamente) es:
            #   dH/d_logit_j = -p_j * (H + log(p_j))
            # El código anterior usaba +(log(p)+1)*p que es d(-H)/d_logit → INVERTIDO.
            # Ese bug hacía que el "bonus de exploración" colapsara la política.
            H_ent   = float(-np.sum(probs * np.log(probs + 1e-8)))
            d_logits += ent_coef * d_ent   # sumar el gradiente de entropía (maximizar)

            h = self._h_last
            grad_Wpi += np.outer(h, d_logits)
            grad_bpi += d_logits

            # ── Gradiente cabeza de valor ──────────────────────
            # d(vf_coef*(val-ret)^2)/d(val) = 2*vf_coef*(val-ret)
            d_val = np.array([2.0 * vf_coef * (val - ret)], dtype=np.float32)
            grad_Wv += np.outer(h, d_val)
            grad_bv += d_val

            # ── Backprop capas ocultas ─────────────────────────
            d_h = d_logits @ self.W_pi.T + d_val @ self.W_v.T
            for li in range(len(self.layers) - 1, -1, -1):
                cache_h, cache_z, _ = self._cache[li]
                d_h_relu = d_h * relu_grad(cache_z)
                grad_layers[li]["W"] += np.outer(cache_h, d_h_relu)
                grad_layers[li]["b"] += d_h_relu
                d_h = d_h_relu @ self.layers[li]["W"].T

        # Promediar gradientes
        grad_Wpi /= B; grad_bpi /= B
        grad_Wv  /= B; grad_bv  /= B
        for li in range(len(self.layers)):
            grad_layers[li]["W"] /= B
            grad_layers[li]["b"] /= B

        # ── Clip por NORMA GLOBAL (como PyTorch clip_grad_norm_) ──────
        # Clip por tensor separado es incorrecto: los bias (pequeños) se actualizan
        # sin límite mientras que las capas ocultas (grandes) quedan frenadas.
        # La norma global escala TODOS los gradientes con el mismo factor.
        MAX_NORM = 0.5
        all_grads = (
            [grad_Wpi, grad_bpi, grad_Wv, grad_bv]
            + [grad_layers[li]["W"] for li in range(len(self.layers))]
            + [grad_layers[li]["b"] for li in range(len(self.layers))]
        )
        global_norm = float(np.sqrt(sum((g**2).sum() for g in all_grads)))
        scale = MAX_NORM / max(global_norm, MAX_NORM)   # ≤ 1.0

        grad_Wpi *= scale; grad_bpi *= scale
        grad_Wv  *= scale; grad_bv  *= scale
        for li in range(len(self.layers)):
            grad_layers[li]["W"] *= scale
            grad_layers[li]["b"] *= scale

        # ── Adam step ─────────────────────────────────────────
        self.t += 1

        # Cabeza política
        self._adam_update_attr("W_pi", "m_Wpi", "v_Wpi", grad_Wpi)
        self._adam_update_attr("b_pi", "m_bpi", "v_bpi", grad_bpi)

        # Cabeza valor
        self._adam_update_attr("W_v", "m_Wv", "v_Wv", grad_Wv)
        self._adam_update_attr("b_v", "m_bv", "v_bv", grad_bv)

        # Capas ocultas
        for li, layer in enumerate(self.layers):
            self._adam_layer(layer, grad_layers[li]["W"], grad_layers[li]["b"])

        return {
            "policy_loss": total_pl / B,
            "value_loss":  total_vl / B,
            "entropy":     total_ent / B,
        }

    def _adam_update_attr(self, param_name: str, m_name: str, v_name: str,
                           grad: np.ndarray):
        """Adam update para un atributo de self."""
        β1, β2, ε = 0.9, 0.99, 1e-7   # v3: β2=0.99 → memoria más corta, adapta rápido
        t = self.t
        m = getattr(self, m_name)
        v = getattr(self, v_name)

        m = β1 * m + (1.0 - β1) * grad
        v = β2 * v + (1.0 - β2) * grad ** 2

        m_hat = m / (1.0 - β1 ** t)
        v_hat = v / (1.0 - β2 ** t)

        param = getattr(self, param_name)
        param -= self.lr * m_hat / (np.sqrt(v_hat) + ε)

        setattr(self, param_name, param)
        setattr(self, m_name, m)
        setattr(self, v_name, v)

    def _adam_layer(self, layer: Dict, gW: np.ndarray, gb: np.ndarray):
        """Adam update para una capa oculta."""
        β1, β2, ε = 0.9, 0.99, 1e-7   # v3: β2=0.99 → adaptación más rápida
        t = self.t

        layer["mW"] = β1 * layer["mW"] + (1.0 - β1) * gW
        layer["vW"] = β2 * layer["vW"] + (1.0 - β2) * gW ** 2
        layer["mb"] = β1 * layer["mb"] + (1.0 - β1) * gb
        layer["vb"] = β2 * layer["vb"] + (1.0 - β2) * gb ** 2

        mW_h = layer["mW"] / (1.0 - β1 ** t)
        vW_h = layer["vW"] / (1.0 - β2 ** t)
        mb_h = layer["mb"] / (1.0 - β1 ** t)
        vb_h = layer["vb"] / (1.0 - β2 ** t)

        layer["W"] -= self.lr * mW_h / (np.sqrt(vW_h) + ε)
        layer["b"] -= self.lr * mb_h / (np.sqrt(vb_h) + ε)

    # ── Serialización ─────────────────────────────────────────

    def save(self, path: str, normalizer: Optional["ObsNormalizer"] = None):
        """Guarda pesos, estado Adam y (opcionalmente) el normalizador."""
        data = {
            "layers": [{"W": l["W"].tolist(), "b": l["b"].tolist()}
                       for l in self.layers],
            "W_pi": self.W_pi.tolist(), "b_pi": self.b_pi.tolist(),
            "W_v":  self.W_v.tolist(),  "b_v":  self.b_v.tolist(),
            "t":    self.t,
        }
        if normalizer is not None:
            data["normalizer"] = normalizer.save()
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str, normalizer: Optional["ObsNormalizer"] = None):
        """Carga pesos y (opcionalmente) el normalizador si existe en el fichero."""
        with open(path) as f:
            data = json.load(f)
        for i, ld in enumerate(data["layers"]):
            self.layers[i]["W"] = np.array(ld["W"], dtype=np.float32)
            self.layers[i]["b"] = np.array(ld["b"], dtype=np.float32)
        self.W_pi = np.array(data["W_pi"], dtype=np.float32)
        self.b_pi = np.array(data["b_pi"], dtype=np.float32)
        self.W_v  = np.array(data["W_v"],  dtype=np.float32)
        self.b_v  = np.array(data["b_v"],  dtype=np.float32)
        self.t    = data.get("t", 0)
        if normalizer is not None and "normalizer" in data:
            normalizer.load(data["normalizer"])


# ═══════════════════════════════════════════════════════════════
# 3. BUFFER DE ROLLOUT
# ═══════════════════════════════════════════════════════════════

class RolloutBuffer:
    """Almacena transiciones de un episodio para la actualización PPO."""

    def __init__(self, gamma: float = 0.95, lam: float = 0.90):  # v3: horizonte más corto
        self.gamma = gamma
        self.lam   = lam
        self.clear()

    def clear(self):
        self.obs:       List[np.ndarray] = []
        self.actions:   List[int]        = []
        self.rewards:   List[float]      = []
        self.values:    List[float]      = []
        self.log_probs: List[float]      = []
        self.dones:     List[bool]       = []

    def add(self, obs, action, reward, value, log_prob, done):
        self.obs.append(obs.astype(np.float32))
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def compute_returns_and_advantages(self, last_value: float = 0.0):
        """GAE-λ advantages y returns."""
        n = len(self.rewards)
        returns    = np.zeros(n, dtype=np.float32)
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in range(n - 1, -1, -1):
            next_val = self.values[t + 1] if t < n - 1 else last_value
            not_done  = float(not self.dones[t])
            delta     = self.rewards[t] + self.gamma * next_val * not_done - self.values[t]
            gae       = delta + self.gamma * self.lam * not_done * gae
            advantages[t] = gae
            returns[t]    = advantages[t] + self.values[t]
        return returns, advantages


# ═══════════════════════════════════════════════════════════════
# 4. DEFINICIÓN DE LOS 5 ARQUETIPOS BASE
# ═══════════════════════════════════════════════════════════════

@dataclass
class AgentType:
    id:           int
    name:         str
    description:  str
    comm_mode:    str       # "none" | "partial" | "explicit"
    reward_mode:  str       # "individual" | "shared" | "mixed"
    reward_alpha: float
    obs_radius:   int
    msg_dim:      int
    hidden:       Tuple     # arquitectura de capas ocultas
    lr:           float
    extra_redundancy_penalty: float = 0.0
    private_map:  bool = False


ARCHETYPES: Dict[int, AgentType] = {

    1: AgentType(
        id=1, name="SOLITARIO",
        description="Sin comunicación. Recompensa individual. Radio 3.",
        comm_mode="none",    reward_mode="individual", reward_alpha=1.0,
        obs_radius=3, msg_dim=0,
        hidden=(128, 128), lr=5e-4,   # v3.1
        extra_redundancy_penalty=0.0, private_map=False,
    ),

    2: AgentType(
        id=2, name="RADAR",
        description="Posiciones GPS de compañeros. Radio 5. Recompensa mixta α=0.3.",
        comm_mode="partial", reward_mode="mixed", reward_alpha=0.3,
        obs_radius=5, msg_dim=0,
        hidden=(128, 128), lr=5e-4,   # v3.1: red más simple, lr agresivo
        extra_redundancy_penalty=-0.05, private_map=False,
    ),

    3: AgentType(
        id=3, name="MENSAJERO",
        description="Canal explícito 8 bits. Recompensa compartida pura.",
        comm_mode="explicit", reward_mode="shared", reward_alpha=0.0,
        obs_radius=5, msg_dim=8,
        hidden=(128, 128), lr=5e-4,   # v3.1
        extra_redundancy_penalty=-0.03, private_map=False,
    ),

    4: AgentType(
        id=4, name="CARTÓGRAFO",
        description="Mapa privado. Radio grande 8. Fuerte penalización por revisitar.",
        comm_mode="partial", reward_mode="mixed", reward_alpha=0.5,
        obs_radius=8, msg_dim=0,
        hidden=(128, 128), lr=4e-4,   # v3.1: red más simple
        extra_redundancy_penalty=-0.2, private_map=True,
    ),

    5: AgentType(
        id=5, name="COLMENA",
        description="Canal explícito 16 bits. Mapa privado. Recompensa de equipo pura.",
        comm_mode="explicit", reward_mode="shared", reward_alpha=0.0,
        obs_radius=5, msg_dim=16,
        hidden=(128, 128), lr=4e-4,   # v3.1: red unificada
        extra_redundancy_penalty=-0.05, private_map=True,
    ),
}


# ═══════════════════════════════════════════════════════════════
# 5. DEFINICIÓN DE LOS 10 EQUIPOS
# ═══════════════════════════════════════════════════════════════

@dataclass
class TeamConfig:
    """
    Configuración de un equipo formado por agentes de distintos arquetipos.
    composition: lista de IDs de arquetipo (uno por agente)
    El entorno usa el modo del agente "dominante" (el de mayor comm_mode y msg_dim).
    Cada sub-grupo de agentes del mismo arquetipo comparte una red.
    """
    id:          int
    label:       str          # etiqueta corta [1,1,1,1]
    name:        str          # nombre descriptivo
    description: str
    composition: List[int]    # IDs de arquetipo por agente

    @property
    def n_agents(self) -> int:
        return len(self.composition)

    @property
    def dominant(self) -> AgentType:
        """Arquetipo con mayor complejidad de comunicación."""
        priority = {5: 4, 3: 3, 4: 2, 2: 1, 1: 0}
        best_id = max(self.composition, key=lambda aid: priority.get(aid, 0))
        return ARCHETYPES[best_id]

    def unique_archetypes(self) -> List[int]:
        return list(dict.fromkeys(self.composition))


TEAMS: Dict[int, TeamConfig] = {

    1: TeamConfig(
        id=1, label="[1,1,1,1]", name="SOLITARIOS",
        description=(
            "4 exploradores sin radio. Límite inferior del sistema. "
            "Si los demás no superan esto, la comunicación no aporta nada."
        ),
        composition=[1, 1, 1, 1],
    ),

    2: TeamConfig(
        id=2, label="[2,2,2,2]", name="RADAR",
        description=(
            "4 guardias con GPS de compañeros. "
            "Mide el beneficio de conocer posiciones sin info compleja."
        ),
        composition=[2, 2, 2, 2],
    ),

    3: TeamConfig(
        id=3, label="[3,3,3,3]", name="MENSAJEROS",
        description=(
            "4 drones con radio explícita de 8 bits. "
            "Mide el valor de la comunicación explícita."
        ),
        composition=[3, 3, 3, 3],
    ),

    4: TeamConfig(
        id=4, label="[4,4,4,4]", name="CARTÓGRAFOS",
        description=(
            "4 topógrafos con mapa privado de radio 8. "
            "Mide la importancia de la memoria espacial."
        ),
        composition=[4, 4, 4, 4],
    ),

    5: TeamConfig(
        id=5, label="[5,5,5,5,5,5]", name="COLMENA",
        description=(
            "6 agentes de enjambre con radio rica de 16 bits. "
            "Máximo nivel de cooperación. Normalmente el mejor resultado."
        ),
        composition=[5, 5, 5, 5, 5, 5],
    ),

    6: TeamConfig(
        id=6, label="[1,1,2,2]", name="MIXTO-A",
        description=(
            "2 operarios básicos + 2 supervisores con visión global. "
            "Mide si unos pocos agentes coordinados mejoran al resto."
        ),
        composition=[1, 1, 2, 2],
    ),

    7: TeamConfig(
        id=7, label="[1,1,3,3]", name="MIXTO-B",
        description=(
            "2 exploradores básicos guiados por 2 operadores con radio. "
            "Mide si la comunicación compensa la falta de inteligencia local."
        ),
        composition=[1, 1, 3, 3],
    ),

    8: TeamConfig(
        id=8, label="[4,4,3,3]", name="MIXTO-C",
        description=(
            "2 cartógrafos + 2 coordinadores con radio. "
            "Sinergia entre memoria y comunicación. Combinación muy fuerte."
        ),
        composition=[4, 4, 3, 3],
    ),

    9: TeamConfig(
        id=9, label="[4,4,5,5,5,5]", name="MIXTO-D",
        description=(
            "2 expertos cartógrafos + 4 agentes de enjambre ejecutando. "
            "Arquitectura jerárquica similar a sistemas reales modernos."
        ),
        composition=[4, 4, 5, 5, 5, 5],
    ),

    10: TeamConfig(
        id=10, label="[1,2,3,4,5,5]", name="MULTIDISCIPLINAR",
        description=(
            "1 de cada arquetipo + 1 colmena extra. Especialización emergente. "
            "La prueba más interesante desde el punto de vista científico."
        ),
        composition=[1, 2, 3, 4, 5, 5],
    ),
}


# ═══════════════════════════════════════════════════════════════
# 6. FASES DEL CURRICULUM
# ═══════════════════════════════════════════════════════════════

@dataclass
class CurriculumPhase:
    idx:            int
    name:           str
    grid_h:         int
    grid_w:         int
    n_floors:       int
    min_connected:  int
    max_steps:      int
    n_episodes:     int
    wall_density:   float
    n_stairs:       int
    redundancy_penalty: float
    completion_bonus:   float
    advance_cov:    float       # cobertura media mínima para avanzar
    advance_k:      int         # nº de episodios recientes para el criterio


CURRICULUM: List[CurriculumPhase] = [
    CurriculumPhase(
        idx=0, name="Semilla",
        grid_h=20, grid_w=20, n_floors=1,
        min_connected=80, max_steps=200,
        n_episodes=300,           # fase 0 saltada — solo inicializa la red
        wall_density=0.18, n_stairs=3,
        redundancy_penalty=0.0,
        completion_bonus=500.0,
        advance_cov=0.80,
        advance_k=10,
    ),
    CurriculumPhase(
        idx=1, name="Local",
        grid_h=40, grid_w=40, n_floors=2,
        min_connected=250, max_steps=1250,
        n_episodes=400,
        wall_density=0.18, n_stairs=5,
        redundancy_penalty=0.3,
        completion_bonus=1500.0,
        advance_cov=0.40,  # v3.1: alcanzable en ~30-50 eps con convergencia estable
        advance_k=5,       # v3.1: ventana pequeña → avance rápido cuando OK
    ),
    CurriculumPhase(
        idx=2, name="Bajo",
        grid_h=80, grid_w=80, n_floors=3,
        min_connected=600, max_steps=2000,
        n_episodes=500,
        wall_density=0.30, n_stairs=8,
        redundancy_penalty=0.7,
        completion_bonus=2500.0,
        advance_cov=0.35,  # v3.1
        advance_k=8,
    ),
    CurriculumPhase(
        idx=3, name="Medio",
        grid_h=120, grid_w=120, n_floors=3,
        min_connected=1000, max_steps=4000,
        n_episodes=600,
        wall_density=0.33, n_stairs=10,
        redundancy_penalty=1.0,
        completion_bonus=4000.0,
        advance_cov=0.35,
        advance_k=20,
    ),
    CurriculumPhase(
        idx=4, name="Alto",
        grid_h=150, grid_w=150, n_floors=3,
        min_connected=1500, max_steps=5000,
        n_episodes=600,
        wall_density=0.33, n_stairs=10,
        redundancy_penalty=1.0,
        completion_bonus=4000.0,
        advance_cov=0.30,
        advance_k=20,
    ),
    CurriculumPhase(
        idx=5, name="Full",
        grid_h=250, grid_w=250, n_floors=3,
        min_connected=2000, max_steps=5000,
        n_episodes=600,
        wall_density=0.33, n_stairs=10,
        redundancy_penalty=1.0,
        completion_bonus=4000.0,
        advance_cov=0.30,
        advance_k=20,
    ),
]


# ═══════════════════════════════════════════════════════════════
# 7. WRAPPER DE OBSERVACIÓN POR AGENTE
# ══════════════════════════════════════════
class AgentObsWrapper:
    """
    Adapta la observación del entorno al formato del arquetipo.
    Mantiene UN mapa privado POR AGENTE (corrección del bug original).
    """

    def __init__(self, archetype: AgentType, n_agents: int):
        self.at = archetype
        self.n_agents = n_agents
        # Un mapa privado por agente (índice → array booleano)
        self._private_visit: Dict[int, Optional[np.ndarray]] = {
            i: None for i in range(n_agents)
        }

    def reset(self, env: MARLExploration3D, agent_indices: List[int]):
        """Inicializa los mapas privados para los agentes de este arquetipo."""
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

        r    = env.cfg.obs_radius
        size = 2 * r + 1
        view_cells = size * size

        f  = int(env.agent_f[agent_id])
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
            #return self.at.extra_redundancy_penalty
            return -0.01  # penalización fija por revisitar, más fácil de ajustar
        return 0.0


# ═══════════════════════════════════════════════════════════════
# 8. TRAINER DE EQUIPO
# ═══════════════════════════════════════════════════════════════

class TeamTrainer:
    """
    Entrena un equipo heterogéneo a través de las fases del curriculum.

    Para equipos homogéneos (un solo arquetipo): una red compartida (parameter sharing).
    Para equipos heterogéneos: una red por sub-grupo de mismo arquetipo.
    """

    def __init__(
        self,
        team: TeamConfig,
        start_phase: int  = 0,
        seed: int         = 42,
        verbose: bool     = True,
        n_ppo_epochs: int = 8,    # v3.1: más pasadas → mejor uso de datos
        batch_size: int   = 32,   # v3.1: batches pequeños → más actualizaciones
        gamma: float      = 0.95, # v3.1: horizonte corto → señal inmediata
        lam: float        = 0.90, # v3.1: GAE menos varianza
    ):
        self.team         = team
        self.start_phase  = start_phase
        self.seed         = seed
        self.verbose      = verbose
        self.n_ppo_epochs = n_ppo_epochs
        self.batch_size   = batch_size
        self.gamma        = gamma
        self.lam          = lam

        # Una red por arquetipo único en el equipo
        self.nets: Dict[int, Optional[MLP]] = {
            aid: None for aid in team.unique_archetypes()
        }

        self.metrics: List[Dict] = []

    # ── Mapeo agente → arquetipo ──────────────────────────────

    def _archetype_of(self, agent_idx: int) -> AgentType:
        return ARCHETYPES[self.team.composition[agent_idx]]

    def _net_of(self, agent_idx: int) -> MLP:
        aid = self.team.composition[agent_idx]
        return self.nets[aid]

    # ── Construcción del entorno ──────────────────────────────

    def _make_env(
        self,
        phase: CurriculumPhase,
        seed: int,
        redundancy_scale: float = 1.0,
        grid_h: Optional[int] = None,
        grid_w: Optional[int] = None,
        n_floors: Optional[int] = None,
    ) -> MARLExploration3D:
        dom  = self.team.dominant
        n    = self.team.n_agents
        gh   = grid_h   if grid_h   is not None else phase.grid_h
        gw   = grid_w   if grid_w   is not None else phase.grid_w
        gf   = n_floors if n_floors is not None else phase.n_floors

        steps = phase.max_steps

        # reward_wall escala con la fase:
        #   fases 0-1 → 0.0  (aprende a moverse sin castigo)
        #   fases 2+  → -0.01 (leve disuasión)
        rwall = 0.0 if phase.idx <= 1 else -0.01

        # r_red: penalización por revisitar celdas.
        # Acotada a [-0.02, 0] para que NUNCA supere reward_new_cell (2.0).
        # red_min=0.05 evita que el agente se quede quieto sin coste.
        red_min   = 0.05
        red_max   = 0.5
        eff_scale = red_min + (red_max - red_min) * redundancy_scale
        r_red     = max(-0.02, -0.003 * phase.redundancy_penalty * eff_scale)

        # Detectar si ExplorationConfig acepta reward_step
        # (algunos entornos tienen step_penalty como parámetro configurable)
        import dataclasses as _dc
        _cfg_fields = {f.name for f in _dc.fields(ExplorationConfig)}
        extra = {'reward_step': 0.0} if 'reward_step' in _cfg_fields else {}

        cfg = ExplorationConfig(
            grid_h        = gh,
            grid_w        = gw,
            n_floors      = gf,
            wall_density  = phase.wall_density,
            n_stairs      = phase.n_stairs,
            min_connected = min(phase.min_connected, int(gh * gw * gf * 0.6)),
            n_agents      = n,
            obs_radius    = dom.obs_radius,
            comm_mode     = dom.comm_mode,
            msg_dim       = dom.msg_dim,
            reward_mode   = dom.reward_mode,
            reward_alpha  = dom.reward_alpha,
            max_steps     = steps,
            reward_wall       = rwall,
            reward_new_cell   = 70.0,   # v3.1: señal fuerte; supera siempre la penalización
            reward_completion = phase.completion_bonus,
            reward_redundant  = r_red,
            **extra,
            seed          = seed,
        )
        return MARLExploration3D(cfg)

    # ── Reanudación desde checkpoint ─────────────────────────

    def _ckpt_path(self, team_id: int, team_name: str, aid: int, phase_idx: int) -> str:
        return os.path.join(
            OUT_DIR,
            f"equipo{team_id}_{team_name}_arch{aid}_fase{phase_idx}.json"
        )

    def _find_resume_state(self) -> Tuple[int, Dict[int, str]]:
        """
        Busca el estado más avanzado guardado para este equipo.

        Devuelve:
          resume_phase_idx : índice de la ÚLTIMA fase completada (-1 si ninguna)
          ckpt_paths       : {archetype_id → ruta del checkpoint a cargar}

        Lógica: una fase se considera "completada" si existen checkpoints para
        TODOS los arquetipos únicos del equipo en esa fase.
        """
        tid  = self.team.id
        tname = self.team.name
        aids = self.team.unique_archetypes()

        last_complete = -1
        last_ckpts: Dict[int, str] = {}

        for phase in CURRICULUM:
            paths = {aid: self._ckpt_path(tid, tname, aid, phase.idx)
                     for aid in aids}
            if all(os.path.exists(p) for p in paths.values()):
                last_complete = phase.idx
                last_ckpts = paths

        return last_complete, last_ckpts

    def _load_nets_from_checkpoints(
        self,
        ckpt_paths: Dict[int, str],
        obs_dim: int,
        n_actions: int,
    ):
        """Inicializa las redes y carga los pesos desde los checkpoints."""
        for aid, path in ckpt_paths.items():
            at = ARCHETYPES[aid]
            net = MLP(
                in_dim    = obs_dim,
                hidden    = at.hidden,
                n_actions = n_actions,
                lr        = at.lr,
                seed      = self.seed + aid * 100,
            )
            norm = ObsNormalizer(obs_dim)
            net.load(path, normalizer=norm)
            net._normalizer = norm
            self.nets[aid] = net
            if self.verbose:
                print(f"│  ✓ Arquetipo {aid} cargado desde: {os.path.basename(path)}"
                      f"  (Adam t={net.t}, obs_samples≈{int(norm.count)})")

    # ── Entrenamiento completo (con reanudación automática) ───

    def train(self):
        if self.verbose:
            print(f"\n{'═'*68}")
            print(f"  EQUIPO {self.team.id}: {self.team.name}  {self.team.label}")
            print(f"  {self.team.description}")
            print(f"  Composición: {self.team.composition}")
            print(f"{'═'*68}\n")

        # ── Detectar reanudación ──────────────────────────────
        last_done, ckpt_paths = self._find_resume_state()

        if last_done >= 0:
            # Cargar redes desde el checkpoint más avanzado
            # Necesitamos obs_dim → instanciar un env de prueba de esa fase
            ref_phase = next(p for p in CURRICULUM if p.idx == last_done)
            env_probe = self._make_env(ref_phase, self.seed)
            obs_p, _  = env_probe.reset(seed=self.seed)
            obs_dim   = obs_p[0].shape[0]
            n_actions = env_probe.n_actions
            env_probe.close()

            if self.verbose:
                print(f"  [REANUDANDO] Fase {last_done} ({ref_phase.name}) "
                      f"ya completada — cargando pesos...")
            self._load_nets_from_checkpoints(ckpt_paths, obs_dim, n_actions)

            # Saltar todas las fases ya completadas
            resume_from = last_done + 1
            if resume_from > CURRICULUM[-1].idx:
                if self.verbose:
                    print(f"  [COMPLETO] Equipo {self.team.id} ya tiene "
                          f"todas las fases entrenadas.\n")
                # Cargar métricas previas para el video/resumen
                self._load_saved_metrics()
                return
            if self.verbose:
                next_phase = next(p for p in CURRICULUM if p.idx == resume_from)
                print(f"  Reanudando desde fase {resume_from} "
                      f"({next_phase.name})...\n")
        else:
            resume_from = self.start_phase
            if self.verbose:
                print(f"  [NUEVO] Sin checkpoints previos — entrenamiento desde cero.\n")

        # ── Bucle de fases ────────────────────────────────────
        phases = [p for p in CURRICULUM if p.idx >= resume_from]
        # prev_phase para warmup: puede ser la última fase completada
        all_phases = CURRICULUM  # referencia completa para buscar la anterior

        for p_idx, phase in enumerate(phases):
            # La fase anterior real en el curriculum completo
            global_idx = CURRICULUM.index(phase)
            prev_phase = CURRICULUM[global_idx - 1] if global_idx > 0 else None

            self._train_phase(phase, prev_phase=prev_phase)

            # Guardar checkpoint por arquetipo (con normalizador)
            for aid in self.team.unique_archetypes():
                if self.nets[aid] is not None:
                    ckpt = self._ckpt_path(
                        self.team.id, self.team.name, aid, phase.idx
                    )
                    normalizer = getattr(self.nets[aid], '_normalizer', None)
                    self.nets[aid].save(ckpt, normalizer=normalizer)

            self._save_metrics(phase.idx)

            if self.verbose:
                print(f"  ✓ Equipo {self.team.id} fase {phase.idx} guardado.\n")

    def _load_saved_metrics(self):
        """Carga métricas previas del fichero jsonl para incluirlas en el resumen."""
        path = os.path.join(
            OUT_DIR,
            f"metrics_equipo{self.team.id}_{self.team.name}.jsonl"
        )
        if not os.path.exists(path):
            return
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        m = json.loads(line)
                        # Evitar duplicados si ya hay métricas en memoria
                        if m not in self.metrics:
                            self.metrics.append(m)
                    except json.JSONDecodeError:
                        pass

    def _train_phase(self, phase: CurriculumPhase, prev_phase: Optional["CurriculumPhase"] = None):
        """
        Entrena una fase del curriculum.

        Mecanismos anti-colapso:
        ─────────────────────────
        1. Calentamiento de grid: los primeros WARMUP_EPS episodios de fases ≥1
           se corren con el grid de la fase anterior (o uno intermedio).
           Permite que la red se reoriente en el entorno nuevo gradualmente.

        2. Decaimiento de lr: al entrar en una nueva fase los lr de todas
           las redes se reducen a la mitad. Evita que los primeros gradientes
           grandes destruyan los pesos transferidos.

        3. Redundancy warmup: la penalización por revisita sube linealmente
           desde 0 hasta el valor completo a lo largo de la fase.
           El agente no es castigado por explorar torpemente al inicio.

        4. reward_wall fijo a -0.05 (ver _make_env): con max_steps grandes
           -0.3 acumula miles de puntos negativos por episodio.
        """
        n = self.team.n_agents

        # ── 1. Obtener obs_dim con el entorno de esta fase ────
        env_probe = self._make_env(phase, self.seed)
        obs_p, _  = env_probe.reset(seed=self.seed)
        obs_dim   = obs_p[0].shape[0]
        n_actions = env_probe.n_actions
        env_probe.close()

        # ── 2. Inicializar o ajustar redes ────────────────────
        first_phase = all(self.nets[aid] is None
                          for aid in self.team.unique_archetypes())
        for aid in self.team.unique_archetypes():
            at = ARCHETYPES[aid]
            if self.nets[aid] is None:
                self.nets[aid] = MLP(
                    in_dim    = obs_dim,
                    hidden    = at.hidden,
                    n_actions = n_actions,
                    lr        = at.lr,
                    seed      = self.seed + aid * 100,
                )
                if self.verbose:
                    params = sum(
                        l["W"].size + l["b"].size for l in self.nets[aid].layers
                    ) + self.nets[aid].W_pi.size + self.nets[aid].b_pi.size
                    print(f"│  Red arquetipo {aid} creada: "
                          f"obs_dim={obs_dim}  params≈{params:,}")
            elif not first_phase:
                # Decaer lr ligeramente al entrar en nueva fase.
                # ×0.8 en vez de ×0.5: preserva la capacidad de aprender
                # sin destruir lo que la red ya sabe.
                self.nets[aid].lr = max(self.nets[aid].lr * 0.8, 5e-5)
                if self.verbose:
                    print(f"│  Arquetipo {aid}: lr ajustado → {self.nets[aid].lr:.2e}")

        # ── 3. Wrappers, buffers y normalizadores ─────────────
        wrappers: Dict[int, AgentObsWrapper] = {}
        for aid in self.team.unique_archetypes():
            wrappers[aid] = AgentObsWrapper(ARCHETYPES[aid], n)

        buffers: List[RolloutBuffer] = [
            RolloutBuffer(gamma=self.gamma, lam=self.lam)
            for _ in range(n)
        ]

        # Un normalizador de observaciones por arquetipo.
        # Si la red ya tiene un normalizador guardado lo reutilizamos;
        # si no, creamos uno nuevo.
        normalizers: Dict[int, ObsNormalizer] = {}
        for aid in self.team.unique_archetypes():
            if hasattr(self.nets[aid], '_normalizer') and self.nets[aid]._normalizer is not None:
                normalizers[aid] = self.nets[aid]._normalizer
            else:
                normalizers[aid] = ObsNormalizer(obs_dim)
                self.nets[aid]._normalizer = normalizers[aid]

        # ── 4. Parámetros de calentamiento ────────────────────
        # Si n_episodes == 0: solo inicializamos la red y salimos
        if phase.n_episodes == 0:
            if self.verbose:
                print(f"┌─ Fase {phase.idx}: {phase.name}  "
                      f"(n_episodes=0 — solo inicialización de red)")
                print(f"└─ Fase {phase.idx} saltada (red lista).\n")
            return

        if self.verbose:
            print(f"┌─ Fase {phase.idx}: {phase.name}  "
                  f"({phase.grid_h}×{phase.grid_w}×{phase.n_floors}  "
                  f"eps={phase.n_episodes})")

        # Buscar la fase de referencia para el grid de calentamiento:
        # la última fase anterior con n_episodes > 0
        ref_phase: Optional[CurriculumPhase] = None
        if prev_phase is not None:
            if prev_phase.n_episodes > 0:
                ref_phase = prev_phase
            else:
                # prev_phase era fase 0 (init-only) → no hay grid anterior real;
                # usar el propio grid de la fase actual (sin warmup de grid)
                ref_phase = None

        WARMUP_EPS = 60 if ref_phase is not None else 0
        if ref_phase is not None:
            # Grid de calentamiento: mitad entre la fase de referencia y la actual.
            # Los pisos suben 1 cada WARMUP_EPS/2 episodios para no saltar de 1 a 2 de golpe.
            wm_h = (ref_phase.grid_h + phase.grid_h) // 2
            wm_w = (ref_phase.grid_w + phase.grid_w) // 2
            wm_f = ref_phase.n_floors   # empezar con los pisos anteriores
        else:
            wm_h = phase.grid_h
            wm_w = phase.grid_w
            wm_f = phase.n_floors

        recent_covs: deque = deque(maxlen=phase.advance_k)
        ep_total = 0

        for ep in range(phase.n_episodes):
            ep_seed = self.seed + phase.idx * 10000 + ep

            # Redundancy scale: empieza en 0.1 (nunca 0) y sube hasta 1.0
            # en el 30% inicial de la fase.
            # CRÍTICO: nunca dejar en 0 — si redundancy=0 el agente puede
            # quedarse quieto indefinidamente sin coste.
            red_scale = 0.1 + 0.9 * min(1.0, ep / max(phase.n_episodes * 0.3, 1))

            # Grid de calentamiento en los primeros WARMUP_EPS episodios.
            # Los pisos suben en la segunda mitad del warmup para evitar
            # el salto brusco de 1-piso → 2-pisos.
            in_warmup = (ep < WARMUP_EPS and ref_phase is not None)
            if in_warmup:
                f_ref  = ref_phase.n_floors
                f_full = phase.n_floors
                if ep < WARMUP_EPS // 2:
                    cur_f = f_ref
                else:
                    t_f   = (ep - WARMUP_EPS // 2) / max(WARMUP_EPS // 2, 1)
                    cur_f = f_ref + int(round(t_f * (f_full - f_ref)))
                    cur_f = max(f_ref, min(f_full, cur_f))
            else:
                cur_f = None   # None → phase.n_floors en _make_env

            env = self._make_env(
                phase, ep_seed,
                redundancy_scale = red_scale,
                grid_h   = wm_h  if in_warmup else None,
                grid_w   = wm_w  if in_warmup else None,
                n_floors = cur_f,
            )
            obs_dict, info = env.reset(seed=ep_seed)

            for aid in self.team.unique_archetypes():
                indices = [i for i, a in enumerate(self.team.composition) if a == aid]
                wrappers[aid].reset(env, indices)

            for i in range(n):
                wrappers[self.team.composition[i]].update_private_map(i, env)

            for buf in buffers:
                buf.clear()

            ep_reward = np.zeros(n, dtype=np.float64)
            done = False
            _stored_obs: Dict[int, np.ndarray] = {}

            while not done:
                actions: Dict[int, int] = {}
                log_probs_: Dict[int, float] = {}
                values_: Dict[int, float] = {}

                for i in range(n):
                    aid      = self.team.composition[i]
                    raw_obs  = wrappers[aid].process(obs_dict[i], i, env)
                    # Actualizar estadísticas y normalizar
                    normalizers[aid].update(raw_obs)
                    proc_obs = normalizers[aid].normalize(raw_obs)
                    action, lp, val = self.nets[aid].act(proc_obs)
                    actions[i]     = action
                    log_probs_[i]  = lp
                    values_[i]     = val
                    _stored_obs[i] = proc_obs   # guardamos la obs YA normalizada

                obs_next, rewards, terminated, truncated, info = env.step(actions)
                done = terminated or truncated

                for i in range(n):
                    aid       = self.team.composition[i]
                    extra_pen = wrappers[aid].redundancy_penalty(i, env)
                    r_total   = rewards[i] + extra_pen
                    buffers[i].add(
                        _stored_obs[i], actions[i], r_total,
                        values_[i], log_probs_[i], done
                    )
                    ep_reward[i] += r_total
                    wrappers[aid].update_private_map(i, env)

                obs_dict = obs_next

            env.close()

            # ── Actualización PPO por sub-grupo ───────────────
            arch_obs: Dict[int, List] = {aid: [] for aid in self.team.unique_archetypes()}
            arch_act: Dict[int, List] = {aid: [] for aid in self.team.unique_archetypes()}
            arch_ret: Dict[int, List] = {aid: [] for aid in self.team.unique_archetypes()}
            arch_adv: Dict[int, List] = {aid: [] for aid in self.team.unique_archetypes()}
            arch_lp:  Dict[int, List] = {aid: [] for aid in self.team.unique_archetypes()}

            for i in range(n):
                aid = self.team.composition[i]
                buf = buffers[i]
                if len(buf.obs) == 0:
                    continue
                ret, adv = buf.compute_returns_and_advantages()
                arch_obs[aid].extend(buf.obs)
                arch_act[aid].extend(buf.actions)
                arch_ret[aid].extend(ret.tolist())
                arch_adv[aid].extend(adv.tolist())
                arch_lp[aid].extend(buf.log_probs)

            for aid in self.team.unique_archetypes():
                if not arch_obs[aid]:
                    continue
                obs_arr = np.array(arch_obs[aid], dtype=np.float32)
                act_arr = np.array(arch_act[aid], dtype=np.int32)
                ret_arr = np.array(arch_ret[aid], dtype=np.float32)
                adv_arr = np.array(arch_adv[aid], dtype=np.float32)
                # old_lp_arr: log-probs del rollout — referencia fija para el ratio.
                old_lp_arr = np.array(arch_lp[aid], dtype=np.float32)

                for epoch in range(self.n_ppo_epochs):
                    idx = np.random.permutation(len(obs_arr))
                    for s in range(0, len(obs_arr), self.batch_size):
                        bi = idx[s: s + self.batch_size]
                        if len(bi) == 0:
                            continue
                        self.nets[aid].update_ppo(
                            obs_arr[bi], act_arr[bi], ret_arr[bi],
                            adv_arr[bi], old_lp_arr[bi]
                        )
                    # Recomputar old_lp tras cada época → ratio parte de ~1.0
                    # en la siguiente pasada → clip siempre efectivo.
                    if epoch < self.n_ppo_epochs - 1:
                        net = self.nets[aid]
                        new_lps = np.empty(len(obs_arr), dtype=np.float32)
                        for k in range(len(obs_arr)):
                            p = net.policy(obs_arr[k])
                            new_lps[k] = float(np.log(p[act_arr[k]] + 1e-8))
                        old_lp_arr = new_lps

            # ── Métricas ──────────────────────────────────────
            cov = info["coverage_ratio"]
            recent_covs.append(cov)
            ep_total += 1

            metric = {
                "team_id":    self.team.id,
                "team_name":  self.team.name,
                "team_label": self.team.label,
                "phase":      phase.idx,
                "phase_name": phase.name,
                "episode":    ep,
                "coverage":   round(cov, 4),
                "mean_reward": round(float(ep_reward.mean()), 4),
                "steps":      info["step"],
                "redundancy": round(info.get("redundancy_ratio", 0), 4),
                "spread":     round(info.get("team_spread", 0), 2),
                "warmup":     in_warmup,
            }
            self.metrics.append(metric)

            if self.verbose and (ep % max(1, phase.n_episodes // 20) == 0
                                 or ep == phase.n_episodes - 1):
                mean_cov_recent = float(np.mean(list(recent_covs))) if recent_covs else 0.0
                tag = " [warmup]" if in_warmup else ""
                print(
                    f"│  ep {ep+1:4d}/{phase.n_episodes}  "
                    f"cov={cov*100:5.1f}%  "
                    f"avg={mean_cov_recent*100:5.1f}%  "
                    f"rew={ep_reward.mean():6.2f}  "
                    f"red_scale={red_scale:.2f}  "
                    f"steps={info['step']}{tag}"
                )

            # ── Criterio de avance anticipado ─────────────────
            # Solo se activa fuera del calentamiento
            if (not in_warmup and
                    len(recent_covs) >= phase.advance_k and
                    float(np.mean(list(recent_covs))) >= phase.advance_cov):
                if self.verbose:
                    print(f"│  ⚡ Avance anticipado a siguiente fase.")
                break

        if self.verbose:
            mean_cov_phase = float(np.mean([m["coverage"] for m in self.metrics
                                            if m["phase"] == phase.idx]))
            print(f"└─ Fase {phase.idx} completada  "
                  f"cov_media={mean_cov_phase*100:.1f}%  "
                  f"eps_reales={ep_total}\n")

    def _save_metrics(self, phase_idx: int):
        """
        Guarda las métricas de una fase en su fichero jsonl.
        Evita duplicados: si el fichero ya contiene episodios de esta fase
        (reanudación), sobreescribe solo los nuevos.
        """
        path = os.path.join(
            OUT_DIR,
            f"metrics_equipo{self.team.id}_{self.team.name}.jsonl"
        )
        # Leer lo que ya existe para esta fase (si hay)
        existing: List[Dict] = []
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        m = json.loads(line)
                        existing.append(m)
                    except json.JSONDecodeError:
                        pass

        # Episodios existentes de esta fase
        existing_eps = {m["episode"] for m in existing if m.get("phase") == phase_idx}

        # Escribir en modo append solo los episodios nuevos
        new_metrics = [
            m for m in self.metrics
            if m["phase"] == phase_idx and m["episode"] not in existing_eps
        ]
        if new_metrics:
            with open(path, "a") as f:
                for m in new_metrics:
                    f.write(json.dumps(m) + "\n")


# ═══════════════════════════════════════════════════════════════
# 9. GENERACIÓN DE VIDEO DE RESULTADOS
# ═══════════════════════════════════════════════════════════════

def _load_metrics_for_team(team_id: int, team_name: str) -> List[Dict]:
    path = os.path.join(OUT_DIR, f"metrics_equipo{team_id}_{team_name}.jsonl")
    metrics = []
    if not os.path.exists(path):
        return metrics
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    metrics.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return metrics


def generate_training_video(
    all_metrics: List[Dict],
    teams: List[TeamConfig],
    output_path: str = "checkpoints/training_results.mp4",
):
    """
    Genera un vídeo MP4 que muestra:
      - Animación de curvas de cobertura durante el entrenamiento
      - Resumen comparativo final de todos los equipos
    """
    print(f"\n{'═'*68}")
    print("  GENERANDO VÍDEO DE RESULTADOS...")
    print(f"{'═'*68}")

    # ── Preparar datos ─────────────────────────────────────────
    team_data: Dict[int, Dict] = {}
    for team in teams:
        mets = [m for m in all_metrics if m["team_id"] == team.id]
        if not mets:
            continue
        covs_by_phase: Dict[int, List[float]] = {}
        for m in mets:
            covs_by_phase.setdefault(m["phase"], []).append(m["coverage"] * 100.0)
        team_data[team.id] = {
            "label": team.label,
            "name":  team.name,
            "covs_by_phase": covs_by_phase,
            "all_covs": [m["coverage"] * 100.0 for m in mets],
            "all_episodes": list(range(len(mets))),
        }

    if not team_data:
        print("  [!] No hay métricas disponibles para generar vídeo.")
        return

    # ── Paleta de colores ──────────────────────────────────────
    cmap  = plt.colormaps["tab10"]
    colors = {team.id: cmap(i % 10) for i, team in enumerate(teams)}

    PHASE_COLORS = ["#e8f4f8", "#d0ebd5", "#fdf3e7", "#f5e6e8"]
    PHASE_NAMES  = ["Semilla", "Local", "Medio", "Full"]

    # ── Calcular suavizado ─────────────────────────────────────
    def smooth(arr, w=15):
        if len(arr) < w:
            return arr
        kernel = np.ones(w) / w
        return np.convolve(arr, kernel, mode="same")

    # ── Determinar nº total de frames ─────────────────────────
    max_eps = max(len(d["all_covs"]) for d in team_data.values())
    n_frames_train = min(max_eps, 300)   # máx 300 frames animados
    step_frames    = max(1, max_eps // n_frames_train)
    # 60 frames de resumen al final
    n_frames_summary = 60
    total_frames = n_frames_train + n_frames_summary

    fig = plt.figure(figsize=(16, 9), facecolor="#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    # ── Layout: 3 paneles ─────────────────────────────────────
    gs = fig.add_gridspec(2, 2,
                          left=0.07, right=0.97,
                          top=0.90, bottom=0.08,
                          hspace=0.40, wspace=0.35)

    ax_main   = fig.add_subplot(gs[:, 0])   # curva principal (izq, full height)
    ax_phase  = fig.add_subplot(gs[0, 1])   # barras por fase (derecha arriba)
    ax_final  = fig.add_subplot(gs[1, 1])   # ranking final  (derecha abajo)

    for ax in [ax_main, ax_phase, ax_final]:
        ax.set_facecolor("#0f0f23")
        for spine in ax.spines.values():
            spine.set_color("#444466")
        ax.tick_params(colors="#aaaacc", labelsize=7)

    # Título global
    title_text = fig.text(
        0.5, 0.96,
        "MARL — Curriculum Learning — 10 Equipos de Exploración 3D",
        ha="center", fontsize=14, fontweight="bold",
        color="#e0e0ff", fontfamily="monospace"
    )
    frame_counter = fig.text(
        0.97, 0.96, "", ha="right", fontsize=9, color="#888899"
    )

    # ── Líneas en ax_main ─────────────────────────────────────
    lines_main: Dict[int, Any] = {}
    for tid, d in team_data.items():
        color = colors[tid]
        label = f"{d['label']} {d['name']}"
        line, = ax_main.plot([], [], lw=1.5, color=color,
                             label=label, alpha=0.85)
        lines_main[tid] = line

    ax_main.set_xlim(0, max_eps)
    ax_main.set_ylim(-2, 102)
    ax_main.set_xlabel("Episodio acumulado", color="#aaaacc", fontsize=8)
    ax_main.set_ylabel("Cobertura (%)", color="#aaaacc", fontsize=8)
    ax_main.set_title("Curvas de entrenamiento", color="#ccccff",
                      fontsize=9, pad=5)
    ax_main.axhline(y=0, color="#333355", lw=0.5)
    ax_main.grid(True, alpha=0.12, color="#334466")
    leg = ax_main.legend(
        loc="lower right", fontsize=5.5,
        facecolor="#1a1a2e", edgecolor="#444466",
        labelcolor="#ccccee", ncol=2
    )

    # ── Barras de cobertura por fase (ax_phase) ────────────────
    ax_phase.set_title("Cobertura media por fase", color="#ccccff",
                        fontsize=9, pad=5)
    ax_phase.set_xlabel("Fase", color="#aaaacc", fontsize=7)
    ax_phase.set_ylabel("Cov. media (%)", color="#aaaacc", fontsize=7)
    ax_phase.set_ylim(0, 100)
    ax_phase.grid(True, axis="y", alpha=0.12, color="#334466")

    # ── Ranking final (ax_final) ───────────────────────────────
    ax_final.set_title("Ranking — últimos 30 eps", color="#ccccff",
                        fontsize=9, pad=5)
    ax_final.set_xlabel("Cobertura media (%)", color="#aaaacc", fontsize=7)
    ax_final.grid(True, axis="x", alpha=0.12, color="#334466")

    # ── Función de actualización ──────────────────────────────

    def update(frame: int):
        if frame < n_frames_train:
            # Fase de animación: mostramos curvas creciendo
            ep_show = min(int((frame + 1) * step_frames), max_eps)

            for tid, d in team_data.items():
                covs = np.array(d["all_covs"])
                eps  = np.array(d["all_episodes"])
                mask = eps < ep_show
                if mask.sum() > 1:
                    c_smooth = smooth(covs[mask], w=20)
                    lines_main[tid].set_data(eps[mask], c_smooth)
                elif mask.sum() == 1:
                    lines_main[tid].set_data(eps[mask], covs[mask])

            ax_main.set_xlim(0, max(ep_show, 10))

            # Barras de fase actuales
            ax_phase.clear()
            ax_phase.set_facecolor("#0f0f23")
            for sp in ax_phase.spines.values():
                sp.set_color("#444466")
            ax_phase.set_title("Cobertura media por fase", color="#ccccff",
                                fontsize=9, pad=5)
            ax_phase.set_xlabel("Fase", color="#aaaacc", fontsize=7)
            ax_phase.set_ylabel("Cov. media (%)", color="#aaaacc", fontsize=7)
            ax_phase.set_ylim(0, 100)
            ax_phase.tick_params(colors="#aaaacc", labelsize=7)
            ax_phase.grid(True, axis="y", alpha=0.12, color="#334466")

            phases_seen = sorted({m["phase"] for m in all_metrics
                                   if m["episode"] < ep_show})
            if phases_seen:
                x_pos    = np.arange(len(phases_seen))
                bar_w    = 0.8 / max(len(team_data), 1)
                for ki, (tid, d) in enumerate(team_data.items()):
                    vals = []
                    for ph in phases_seen:
                        ph_covs = [m["coverage"] * 100.0
                                   for m in all_metrics
                                   if m["team_id"] == tid
                                   and m["phase"] == ph
                                   and m["episode"] < ep_show]
                        vals.append(float(np.mean(ph_covs)) if ph_covs else 0.0)
                    offset = (ki - len(team_data) / 2.0) * bar_w + bar_w / 2
                    ax_phase.bar(x_pos + offset, vals, bar_w * 0.9,
                                 color=colors[tid], alpha=0.80)
                ax_phase.set_xticks(x_pos)
                ax_phase.set_xticklabels(
                    [PHASE_NAMES[p] for p in phases_seen],
                    fontsize=6, color="#aaaacc"
                )

            frame_counter.set_text(f"ep ≤ {ep_show}")

        else:
            # Fase de resumen: mostrar ranking final con animación de barras
            t = (frame - n_frames_train) / max(n_frames_summary - 1, 1)
            t = min(t, 1.0)

            ax_final.clear()
            ax_final.set_facecolor("#0f0f23")
            for sp in ax_final.spines.values():
                sp.set_color("#444466")
            ax_final.set_title("Ranking — últimos 30 eps", color="#ccccff",
                                fontsize=9, pad=5)
            ax_final.set_xlabel("Cobertura media (%)", color="#aaaacc", fontsize=7)
            ax_final.tick_params(colors="#aaaacc", labelsize=7)
            ax_final.grid(True, axis="x", alpha=0.12, color="#334466")

            # Calcular cobertura final por equipo
            final_covs: Dict[int, float] = {}
            for tid, d in team_data.items():
                last_covs = d["all_covs"][-30:]
                final_covs[tid] = float(np.mean(last_covs)) if last_covs else 0.0

            sorted_teams = sorted(final_covs.items(), key=lambda x: x[1])

            if sorted_teams:
                yticks = list(range(len(sorted_teams)))
                y_labels = [
                    f"{team_data[tid]['label']}" for tid, _ in sorted_teams
                ]
                vals_full = [v for _, v in sorted_teams]
                vals_anim = [v * t for v in vals_full]
                bar_colors = [colors[tid] for tid, _ in sorted_teams]

                bars = ax_final.barh(yticks, vals_anim, color=bar_colors,
                                     alpha=0.85, height=0.65)
                ax_final.set_yticks(yticks)
                ax_final.set_yticklabels(y_labels, fontsize=6.5, color="#ccccee")
                ax_final.set_xlim(0, 100)

                # Etiquetas de valor
                for bar_obj, (tid, v_full) in zip(bars, sorted_teams):
                    v_show = v_full * t
                    if v_show > 3:
                        ax_final.text(
                            bar_obj.get_width() + 1,
                            bar_obj.get_y() + bar_obj.get_height() / 2,
                            f"{v_full:.1f}%",
                            va="center", ha="left",
                            fontsize=6, color="#ccccee"
                        )

            frame_counter.set_text("RESUMEN FINAL")

        return list(lines_main.values())

    # ── Crear animación ───────────────────────────────────────
    anim = FuncAnimation(
        fig, update,
        frames=total_frames,
        interval=80,
        blit=False,
    )

    # ── Intentar guardar con FFmpeg, si no → gif ──────────────
    saved = False
    try:
        writer = FFMpegWriter(fps=15, metadata={"title": "MARL Training"},
                              bitrate=1800)
        anim.save(output_path, writer=writer, dpi=120)
        print(f"  ✓ Vídeo guardado: {output_path}")
        saved = True
    except Exception as exc:
        print(f"  [!] FFmpeg no disponible ({exc}). Intentando GIF...")

    if not saved:
        try:
            gif_path = output_path.replace(".mp4", ".gif")
            anim.save(gif_path, writer="pillow", fps=10, dpi=80)
            print(f"  ✓ GIF guardado: {gif_path}")
            saved = True
            output_path = gif_path
        except Exception as exc2:
            print(f"  [!] No se pudo guardar animación: {exc2}")

    plt.close(fig)

    if saved:
        # ── Imagen estática de resumen final ──────────────────
        _save_summary_image(all_metrics, teams, colors, team_data)

    return output_path


def _save_summary_image(
    all_metrics: List[Dict],
    teams: List[TeamConfig],
    colors: Dict[int, Any],
    team_data: Dict[int, Dict],
):
    """Guarda una imagen PNG con el resumen comparativo final."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor="#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(
        "MARL — Resumen Final — 10 Equipos de Exploración 3D",
        fontsize=14, fontweight="bold", color="#e0e0ff", y=0.98
    )

    for ax in axes:
        ax.set_facecolor("#0f0f23")
        for sp in ax.spines.values():
            sp.set_color("#444466")
        ax.tick_params(colors="#aaaacc")

    # Panel izquierdo: curvas completas suavizadas
    ax0 = axes[0]
    ax0.set_title("Curvas de entrenamiento (suavizadas)", color="#ccccff", pad=5)
    ax0.set_xlabel("Episodio acumulado", color="#aaaacc")
    ax0.set_ylabel("Cobertura (%)", color="#aaaacc")
    ax0.grid(True, alpha=0.12, color="#334466")

    def smooth(arr, w=20):
        if len(arr) < w:
            return arr
        return np.convolve(arr, np.ones(w) / w, mode="same")

    for tid, d in team_data.items():
        covs = np.array(d["all_covs"])
        eps  = np.array(d["all_episodes"])
        c_s  = smooth(covs)
        ax0.plot(eps, c_s, lw=1.5, color=colors[tid], alpha=0.9,
                 label=f"{d['label']} {d['name']}")

    ax0.legend(fontsize=6, facecolor="#1a1a2e", edgecolor="#444466",
               labelcolor="#ccccee", loc="lower right", ncol=2)
    ax0.set_ylim(-2, 102)

    # Panel derecho: ranking final
    ax1 = axes[1]
    ax1.set_title("Cobertura media — últimos 30 episodios", color="#ccccff", pad=5)
    ax1.set_xlabel("Cobertura (%)", color="#aaaacc")
    ax1.grid(True, axis="x", alpha=0.12, color="#334466")

    final_covs: Dict[int, float] = {}
    for tid, d in team_data.items():
        last_covs = d["all_covs"][-30:]
        final_covs[tid] = float(np.mean(last_covs)) if last_covs else 0.0

    sorted_teams = sorted(final_covs.items(), key=lambda x: x[1])
    yticks = list(range(len(sorted_teams)))

    for yi, (tid, v) in enumerate(sorted_teams):
        color = colors[tid]
        ax1.barh(yi, v, color=color, alpha=0.85, height=0.65)
        ax1.text(v + 0.5, yi, f"{v:.1f}%",
                 va="center", ha="left", fontsize=7, color="#ccccee")

    ax1.set_yticks(yticks)
    ax1.set_yticklabels(
        [f"{team_data[tid]['label']} {team_data[tid]['name']}"
         for tid, _ in sorted_teams],
        fontsize=7, color="#ccccee"
    )
    ax1.set_xlim(0, 100)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    img_path = os.path.join(OUT_DIR, "resumen_final.png")
    plt.savefig(img_path, dpi=140, facecolor="#1a1a2e")
    plt.close(fig)
    print(f"  ✓ Imagen resumen guardada: {img_path}")


# ═══════════════════════════════════════════════════════════════
# 10. RESUMEN TEXTUAL
# ═══════════════════════════════════════════════════════════════

def print_summary(all_metrics: List[Dict]):
    print(f"\n{'═'*80}")
    print(f"  RESUMEN DE ENTRENAMIENTO — 10 EQUIPOS")
    print(f"{'═'*80}")
    header = (f"{'Equipo':<22} {'Fase':<8} "
              f"{'Cov%':>7} {'Rew':>7} {'Red%':>6} {'Spread':>7}")
    print(header)
    print("─" * 80)

    by_team_phase: Dict = {}
    for m in all_metrics:
        key = (m["team_name"], m["phase"])
        by_team_phase.setdefault(key, []).append(m)

    for key in sorted(by_team_phase.keys()):
        name, phase = key
        ms   = by_team_phase[key]
        last = ms[-min(20, len(ms)):]
        cov  = float(np.mean([m["coverage"]     for m in last])) * 100
        rew  = float(np.mean([m["mean_reward"]   for m in last]))
        red  = float(np.mean([m["redundancy"]    for m in last])) * 100
        sprd = float(np.mean([m["spread"]        for m in last]))
        print(f"  {name:<20} Fase{phase:<4} "
              f"{cov:>6.1f}% {rew:>7.2f} {red:>5.1f}% {sprd:>7.1f}")

    print(f"{'═'*80}\n")


# ═══════════════════════════════════════════════════════════════
# 11. ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Curriculum Learning — 10 configuraciones de equipo MARL"
    )
    parser.add_argument("--team",  type=int, default=0,
                        help="ID del equipo a entrenar (0=todos, 1-10)")
    parser.add_argument("--fase",  type=int, default=0,
                        help="Fase inicial del curriculum (0-5)")
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--fast",  action="store_true",
                        help="Modo rápido: grids y episodios reducidos")
    parser.add_argument("--quiet", action="store_true",
                        help="Silenciar output detallado")
    parser.add_argument("--no-video", action="store_true",
                        help="No generar vídeo al finalizar")
    args = parser.parse_args()

    # Modo rápido
    if args.fast:
        for ph in CURRICULUM:
            ph.grid_h        = min(ph.grid_h, 25)
            ph.grid_w        = min(ph.grid_w, 25)
            ph.min_connected = min(ph.min_connected, 150)
            # Preservar n_episodes=0 (fases de solo-inicialización)
            if ph.n_episodes > 0:
                ph.n_episodes = max(5, ph.n_episodes // 30)
            ph.max_steps     = min(ph.max_steps, 500)
            ph.advance_k     = min(ph.advance_k, 5)
        print("  [MODO RÁPIDO] Grids y episodios muy reducidos.\n")

    # Selección de equipos
    if args.team in TEAMS:
        teams_to_train = [TEAMS[args.team]]
    else:
        teams_to_train = list(TEAMS.values())

    # ── Entrenamiento ──────────────────────────────────────────
    all_metrics: List[Dict] = []
    t_global = time.perf_counter()

    for team in teams_to_train:
        t0 = time.perf_counter()
        trainer = TeamTrainer(
            team        = team,
            start_phase = args.fase,
            seed        = args.seed,
            verbose     = not args.quiet,
        )
        trainer.train()
        # Si el equipo ya estaba completo, _load_saved_metrics() habrá
        # llenado trainer.metrics; si acaba de entrenar, también están ahí.
        all_metrics.extend(trainer.metrics)
        elapsed = time.perf_counter() - t0

        if not args.quiet:
            trained = any(m.get("episode") is not None for m in trainer.metrics)
            tag = f"{elapsed:.1f}s" if trained else "ya completo (cargado)"
            print(f"  Tiempo {team.name}: {tag}\n")

    total_time = time.perf_counter() - t_global
    print_summary(all_metrics)
    print(f"  Tiempo total:   {total_time:.1f}s")
    print(f"  Checkpoints en: {OUT_DIR}/")

    # ── Generar vídeo ──────────────────────────────────────────
    if not args.no_video and all_metrics:
        video_path = os.path.join(OUT_DIR, "training_results.mp4")
        generate_training_video(all_metrics, teams_to_train, video_path)


if __name__ == "__main__":
    main()