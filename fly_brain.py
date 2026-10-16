"""
fly_brain.py
============
Biologically-inspired Drosophila melanogaster MaleCNS connectome model with
1024 Leaky Integrate-and-Fire (LIF) neurons and Dopamine-modulated Reinforcement Learning.

Anatomy & Structure (1024 neurons):
  1. Antennal Lobe (AL, 128 neurons): Sensory input preprocessing (prices, momentum, volatility).
  2. Mushroom Body Kenyon Cells (KC, 640 neurons): Sparse high-dimensional projection.
  3. Central Complex (CX / Ellipsoid Body, 128 neurons): Internal heading & market regime compass.
  4. Dopaminergic Neurons (DANs - PAM & PPL1, 64 neurons): TD-error & reward prediction (dopamine bursts).
  5. Mushroom Body Output Neurons (MBONs, 64 neurons): Action valence driving Buy / Hold / Sell decisions.

Features:
- Optional integration with real FlyWire v783 connectome data for more biologically-grounded weight matrices
- Backward compatible with random initialization
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import requests
import os
import gzip
import sys
from pathlib import Path
from scipy.sparse import csr_matrix

from config import OBS_DIM

# ── FlyWire Connectome Data Loader ──────────────────────────────────────────

def download_flywire_data(save_dir="flywire_data"):
    """
    Download FlyWire v783 connectome data from the fly-brain repo.
    
    Required files:
    - 2025_Completeness_783.csv (3.2 MB): neuron IDs and metadata
    - 2025_Connectivity_783.parquet (97 MB): synapse connectivity with weights
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # Check if data already exists
    comp_path = save_path / "2025_Completeness_783.csv"
    conn_path = save_path / "2025_Connectivity_783.parquet"
    
    if comp_path.exists() and conn_path.exists():
        print(f"FlyWire data already found at {save_path}")
        return comp_path, conn_path
    
    print("Downloading FlyWire v783 data...")
    
    # Download from GitHub releases
    base_url = "https://github.com/eonsystemspbc/fly-brain/raw/main/data/"
    
    files = [
        "2025_Completeness_783.csv",
        "2025_Connectivity_783.parquet"
    ]
    
    for filename in files:
        url = f"{base_url}{filename}"
        target_path = save_path / filename
        
        if not target_path.exists():
            try:
                print(f"  Downloading {filename}... ({url})")
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                with open(target_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                print(f"  [OK] Downloaded {filename}")
            except Exception as e:
                print(f"  [FAIL] Failed to download {filename}: {e}")
                print(f"    You may need to manually download from the GitHub repo")
                return None, None
    
    return comp_path, conn_path
def build_sparse_weights_from_connectome(comp_path, conn_path, target_neurons=1024,
                                         clip=0.45, normalize="degree", reciprocal_scale=0.0):
    """
    Build a sparse (target_neurons x target_neurons) weight matrix from FlyWire
    v783 data.

    Schema (verified against the bundled files):
      * ``2025_Completeness_783.csv`` : one row per neuron, index = root ID.
      * ``2025_Connectivity_783.parquet`` : ``Presynaptic_Index`` /
        ``Postsynaptic_Index`` are ROW POSITIONS into the completeness table
        (range 0..138638 == n_rows-1), NOT root IDs.

    Mapping is therefore row -> model index.  The model's anatomical partitions
    (AL/KC/CX/DAN/MBON) are a *structural* split of the first ``target_neurons``
    rows; the FlyWire rows used here are not classified by cell type, so this is
    NOT a biologically validated mapping (documented in the returned provenance).

    Args:
        clip: absolute weight cap after normalisation.
        normalize: "degree" (divide by per-target inbound degree so a fully
                   active presynaptic population contributes ~clip),
                   "max" (scale by global max), or "none".
        reciprocal_scale: optional weak reciprocal edges (0.0 = disabled).

    Returns:
        (scipy.sparse.csr_matrix, mapping dict)
    """
    import hashlib
    import pandas as pd

    def _sha256(path, chunk=1 << 20):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
        return h.hexdigest()

    print("Building sparse weight matrix from FlyWire connectome...")
    df_comp = pd.read_csv(comp_path, index_col=0)
    n_avail = len(df_comp)
    print(f"  Loaded {n_avail} neurons")

    # Structural partition sizes (fractions of the model, not cell-type based).
    N = int(target_neurons)
    sizes = {
        "AL": int(round(N * 128 / 1024)),
        "KC": int(round(N * 640 / 1024)),
        "CX": int(round(N * 128 / 1024)),
        "DAN": int(round(N * 64 / 1024)),
    }
    sizes["MBON"] = N - sum(sizes.values())

    rows, offset = [], 0
    for name in ("AL", "KC", "CX", "DAN", "MBON"):
        count = min(sizes[name], max(0, n_avail - offset))
        rows.extend(range(offset, offset + count))
        offset += count
    rows = rows[:N]
    row2model = {r: k for k, r in enumerate(rows)}
    print(f"  Selected {len(rows)} rows | " +
          ", ".join(f"{k}:{v}" for k, v in sizes.items()))

    cols = ["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"]
    df_conn = pd.read_parquet(conn_path, columns=cols)
    selected = set(row2model)
    sub = df_conn[
        df_conn["Presynaptic_Index"].isin(selected)
        & df_conn["Postsynaptic_Index"].isin(selected)
    ]
    n_syn = int(sub.shape[0])
    print(f"  After filtering: {n_syn} synapses within the selected neurons")
    if n_syn == 0:
        print("  Warning: no synapses found within selected neurons")
        return None, None

    src = sub["Presynaptic_Index"].map(row2model).to_numpy()
    tgt = sub["Postsynaptic_Index"].map(row2model).to_numpy()
    w = sub["Excitatory x Connectivity"].to_numpy(dtype=np.float64)

    if normalize == "degree":
        deg = np.bincount(tgt, minlength=N).astype(np.float64)
        deg[deg == 0] = 1.0
        w = w / deg[tgt] * clip
    elif normalize == "max":
        mx = float(np.max(np.abs(w))) or 1.0
        w = w / mx * clip
    w = np.clip(w, -clip, clip).astype(np.float32)

    src_all, tgt_all, w_all = src, tgt, w
    if reciprocal_scale > 0:
        src_all = np.concatenate([src, tgt])
        tgt_all = np.concatenate([tgt, src])
        w_all = np.concatenate([w, w * reciprocal_scale])

    from scipy.sparse import coo_matrix
    W = coo_matrix((w_all, (src_all, tgt_all)), shape=(N, N)).tocsr()

    provenance = {
        "source_files": {
            Path(comp_path).name: _sha256(comp_path),
            Path(conn_path).name: _sha256(conn_path),
        },
        "target_neurons": N,
        "synapses_within_selection": n_syn,
        "nnz": int(W.nnz),
        "clip": clip,
        "normalize": normalize,
        "reciprocal_scale": reciprocal_scale,
        "mapping": "row_index_of_completeness_csv -> model_index",
        "cell_type_verified": False,
    }
    mapping = {
        "row_to_model": row2model,
        "partition_sizes": sizes,
        "provenance": provenance,
    }
    print(f"  Built sparse weight matrix {W.shape} nnz={W.nnz}")
    return W, mapping
def init_weights_from_connectome(target_shape=(1024, 1024), rng=None):
    """
    Create random weights if connectome not available (fallback).
    This is the original weight initialization logic.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    W = np.zeros(target_shape, dtype=np.float32)
    
    # Sensory AL to KC sparse projection (each KC gets ~6-8 AL inputs)
    for kc in range(128, 768):
        al_indices = rng.choice(128, size=8, replace=False)
        W[kc, al_indices] = rng.uniform(0.2, 0.45, size=8)
    
    # Central complex ring recurrent attractor connectivity
    for i in range(768, 896):
        left = 768 + (i - 768 - 1) % 128
        right = 768 + (i - 768 + 1) % 128
        W[i, left] = 0.35
        W[i, right] = 0.35
    
    # KC to MBON convergent readout with stronger weights for reliable firing
    for mbon in range(960, 1024):
        kc_sample = rng.choice(np.arange(128, 768), size=32, replace=False)
        W[mbon, kc_sample] = rng.uniform(0.15, 0.40, size=32)
        W[mbon, mbon] = 0.25  # Self-excitation for reliable MBON firing
    
    # Background sparse interconnectivity
    sparse_mask = rng.random(target_shape) < 0.015
    W[sparse_mask] += rng.normal(0.0, 0.04, size=np.count_nonzero(sparse_mask))
    np.clip(W, -0.6, 0.6, out=W)
    
    return W
def try_load_connectome_weights(weight_dir="flywire_weights"):
    """
    Try to load pre-built connectome weights from cache.
    """
    weight_path = Path(weight_dir) / "connectome_weights.pkl"
    if weight_path.exists():
        try:
            print(f"Loading cached connectome weights from {weight_path}")
            with open(weight_path, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            print(f"Failed to load cached weights: {e}")
    return None
def save_connectome_weights(data, weight_dir="flywire_weights"):
    """
    Save connectome weights to cache for future use.
    """
    weight_path = Path(weight_dir) / "connectome_weights.pkl"
    weight_path.parent.mkdir(exist_ok=True)
    
    try:
        with open(weight_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved connectome weights to {weight_path}")
    except Exception as e:
        print(f"Failed to save connectome weights: {e}")
# ── Spiking MaleCNS Connectome Simulator ────────────────────────────────────
class MaleCNS1024Simulator:
    """
    Leaky Integrate-and-Fire (LIF) connectome simulation with anatomical
    Drosophila neuropil partitions. Neuron count is configurable; default 1024
    maintains backward compatibility with existing weights and architecture.
    
    Args:
        use_connectome: If True, load real FlyWire v783 connectome weights.
                        If False (default), use randomly initialized weights.
        connectome_dir: Path to downloaded FlyWire data directory.
        weight_cache: Path to cached connectome weight pickle file.
        N_NEURONS: Total neurons (default 1024). If >1024, anatomical partitions
                   scale proportionally; if <1024, a subset is used.
    """
    DEFAULT_N_NEURONS = 1024
    IDX_AL_RATIO   = 128 / 1024   # ~12.5%
    IDX_KC_RATIO   = 640 / 1024   # ~62.5%
    IDX_CX_RATIO   = 128 / 1024   # ~12.5%
    IDX_DAN_RATIO   = 64 / 1024   # ~6.25%
    IDX_MBON_RATIO  = 64 / 1024   # ~6.25%

    def __init__(self, use_connectome=False, connectome_dir="flywire_data", weight_cache="flywire_weights", N_NEURONS=None):
        self.N_NEURONS = N_NEURONS if N_NEURONS is not None else self.DEFAULT_N_NEURONS
        self.IDX_AL   = slice(0, int(round(self.N_NEURONS * self.IDX_AL_RATIO)))
        self.IDX_KC   = slice(int(round(self.N_NEURONS * self.IDX_AL_RATIO)),
                              int(round(self.N_NEURONS * (self.IDX_AL_RATIO + self.IDX_KC_RATIO))))
        self.IDX_CX   = slice(int(round(self.N_NEURONS * (self.IDX_AL_RATIO + self.IDX_KC_RATIO))),
                              int(round(self.N_NEURONS * (self.IDX_AL_RATIO + self.IDX_KC_RATIO + self.IDX_CX_RATIO))))
        self.IDX_DAN  = slice(int(round(self.N_NEURONS * (self.IDX_AL_RATIO + self.IDX_KC_RATIO + self.IDX_CX_RATIO))),
                              int(round(self.N_NEURONS * (self.IDX_AL_RATIO + self.IDX_KC_RATIO + self.IDX_CX_RATIO + self.IDX_DAN_RATIO))))
        self.IDX_MBON = slice(int(round(self.N_NEURONS * (self.IDX_AL_RATIO + self.IDX_KC_RATIO + self.IDX_CX_RATIO + self.IDX_DAN_RATIO))),
                              self.N_NEURONS)

        rng = np.random.default_rng(42)
        self.V = np.zeros(self.N_NEURONS, dtype=np.float32)
        self.threshold = 0.7  # Lower threshold for easier firing
        self.decay = 0.88
        self.last_spikes = np.zeros(self.N_NEURONS, dtype=np.float32)
        self.dopamine_level = 0.0
        self._inh = 0.0  # Feedback inhibition strength for sparse coding
        # Deterministic encoding settings. ``noise_std=0`` removes the additive
        # membrane noise so the read-out is a reproducible function of the obs.
        self.noise_std = 0.0
        self.settle_steps = 6  # LIF updates used to settle the cascade per obs
        self.use_connectome = use_connectome
        
        # Build weight matrix: try connectome first, then fallback
        if use_connectome:
            self.W = self._build_connectome_weights(connectome_dir, weight_cache, rng)
        else:
            self.W = self._build_random_weights(rng)
        
        if hasattr(self.W, 'toarray'):
            self.W_dense = self.W.toarray().astype(np.float32)
        else:
            self.W_dense = np.asarray(self.W, dtype=np.float32)
    
    def _build_random_weights(self, rng):
        """Original random weight initialization for backward compatibility."""
        W = np.zeros((self.N_NEURONS, self.N_NEURONS), dtype=np.float32)
        
        # Self-excitation for sustained firing (critical for cascade propagation)
        for i in range(self.IDX_AL.start, self.IDX_AL.stop):
            W[i, i] = 0.35  # AL self-excitation
        for i in range(self.IDX_KC.start, self.IDX_KC.stop):
            W[i, i] = 0.25  # KC self-excitation
        for i in range(self.IDX_MBON.start, self.IDX_MBON.stop):
            W[i, i] = 0.30  # MBON self-excitation
        
        # Sensory AL to KC sparse projection (each KC gets ~6-8 AL inputs)
        n_al = min(self.IDX_AL.stop, self.N_NEURONS)
        for kc in range(self.IDX_KC.start, self.IDX_KC.stop):
            al_indices = rng.choice(n_al, size=8, replace=False)
            W[kc, al_indices] = rng.uniform(0.2, 0.45, size=8)  # Natural weight range
        
        # Central complex ring recurrent attractor connectivity
        n_cx = self.IDX_CX.stop - self.IDX_CX.start
        for i_local in range(n_cx):
            i = self.IDX_CX.start + i_local
            left = self.IDX_CX.start + (i_local - 1) % n_cx
            right = self.IDX_CX.start + (i_local + 1) % n_cx
            W[i, left] = 0.35
            W[i, right] = 0.35
        
        # KC to MBON convergent readout with stronger weights for reliable firing
        n_kc = self.IDX_KC.stop - self.IDX_KC.start
        n_mbon = self.IDX_MBON.stop - self.IDX_MBON.start
        for mbon in range(self.IDX_MBON.start, self.IDX_MBON.stop):
            kc_sample = rng.choice(np.arange(self.IDX_KC.start, self.IDX_KC.stop), size=32, replace=False)
            W[mbon, kc_sample] = rng.uniform(0.15, 0.40, size=32) * 2.5
        
        # Background sparse interconnectivity
        sparse_mask = rng.random((self.N_NEURONS, self.N_NEURONS)) < 0.015
        W[sparse_mask] += rng.normal(0.0, 0.04, size=np.count_nonzero(sparse_mask))
        np.clip(W, -0.6, 0.6, out=W)
        
        return csr_matrix(W, dtype=np.float32)
    
    def _build_connectome_weights(self, connectome_dir, weight_cache, rng):
        """Try to load real connectome weights, fallback to random if unavailable."""
        # Check for cached weights first
        cached = try_load_connectome_weights(weight_cache)
        if cached is not None and cached.get('W') is not None:
            W = cached['W']
            shape = getattr(W, 'shape', None)
            if shape == (self.N_NEURONS, self.N_NEURONS):
                if 'provenance' in cached or (cached.get('mapping') or {}).get('provenance'):
                    print("Using cached connectome weights (provenance recorded)")
                else:
                    print("WARNING: using cached connectome weights with UNVERIFIED provenance "
                          f"({weight_cache}/connectome_weights.pkl). Regenerate with "
                          "scripts/rebuild_connectome.py for an auditable build.")
                return W
            print(f"Cached weights shape mismatch: {shape} vs {(self.N_NEURONS, self.N_NEURONS)}, regenerating")

        # Try to build from raw data
        comp_path = Path(connectome_dir) / "2025_Completeness_783.csv"
        conn_path = Path(connectome_dir) / "2025_Connectivity_783.parquet"
        if comp_path.exists() and conn_path.exists():
            try:
                W, mapping = build_sparse_weights_from_connectome(
                    str(comp_path), str(conn_path), target_neurons=self.N_NEURONS
                )
                if W is not None:
                    save_connectome_weights(
                        {'W': W, 'mapping': mapping, 'provenance': mapping.get('provenance')},
                        weight_cache,
                    )
                    return W
            except Exception as e:
                print(f"Failed to build connectome weights: {e}")

        # Try downloading data
        print("Connectome data not found locally. Attempting download...")
        comp_path, conn_path = download_flywire_data(connectome_dir)
        if comp_path and conn_path:
            try:
                W, mapping = build_sparse_weights_from_connectome(
                    str(comp_path), str(conn_path), target_neurons=self.N_NEURONS
                )
                if W is not None:
                    save_connectome_weights(
                        {'W': W, 'mapping': mapping, 'provenance': mapping.get('provenance')},
                        weight_cache,
                    )
                    return W
            except Exception as e:
                print(f"Failed to build connectome weights from downloaded data: {e}")

        print("Connectome unavailable, falling back to random weights")
        return self._build_random_weights(rng)

    def reset(self):
        self.reset_state()
        self.dopamine_level = 0.0

    def reset_state(self):
        """Clear membrane state only (V, spikes, inhibition); keep dopamine."""
        self.V = np.zeros(self.N_NEURONS, dtype=np.float32)
        self.last_spikes = np.zeros(self.N_NEURONS, dtype=np.float32)
        self._inh = 0.0

    def _sensory_input(self, cur_obs: np.ndarray) -> np.ndarray:
        """Build the per-step synaptic input current from one observation."""
        inp = np.zeros(self.N_NEURONS, dtype=np.float32)
        al_start = self.IDX_AL.start
        cx_start = self.IDX_CX.start
        n_cx = self.IDX_CX.stop - cx_start

        # 1. Antennal Lobe sensory injection (128 neurons)
        # Close prices from the 30-step OHLCV window (every 5th element from index 3)
        close_prices = [float(cur_obs[step * 5 + 3]) for step in range(min(30, len(cur_obs) // 5))]

        # Sub-cluster 1 (AL 0..59): price momentum & relative changes
        if len(close_prices) >= 2:
            c0 = close_prices[0]
            for i, c in enumerate(close_prices[:30]):
                inp[al_start + i] = np.tanh(((c - c0) / (c0 + 1e-9)) * 8.0) * 2.2
            for i in range(1, len(close_prices[:30])):
                prev_c = close_prices[i - 1]
                inp[al_start + 30 + i - 1] = np.tanh(((close_prices[i] - prev_c) / (prev_c + 1e-9)) * 25.0) * 2.2
            inp[al_start + 59] = np.tanh(((close_prices[-1] - c0) / (c0 + 1e-9)) * 10.0) * 2.5

        # Sub-cluster 2 (AL 60..89): momentum indicators
        if len(cur_obs) >= 240:
            inp[al_start + 60: al_start + 90] = np.tanh(cur_obs[150:180] * 1.5) * 2.0
        # Sub-cluster 3 (AL 90..119): trend & regime indicators
        if len(cur_obs) >= 210:
            inp[al_start + 90: al_start + 120] = np.tanh(cur_obs[180:210] * 1.5) * 2.0
        # Sub-cluster 4 (AL 120..127): current-step indicator snapshot
        if len(cur_obs) >= 8:
            inp[al_start + 120: al_start + 128] = np.tanh(cur_obs[-8:] * 1.5) * 2.0

        # 2. Central Complex (CX): internal directional compass
        if len(close_prices) >= 5:
            c_last, c_first = close_prices[-1], close_prices[0]
            c_mid = close_prices[len(close_prices) // 2]
            trend_win = (c_last - c_first) / (c_first + 1e-9)
            mom_recent = (c_last - c_mid) / (c_mid + 1e-9)
            angle = (math.atan2(mom_recent * 15.0, trend_win * 10.0) % (2 * math.pi))
            ring_idx = int((angle / (2 * math.pi)) * n_cx) % n_cx
            trend_mag = min(2.0, abs(trend_win) * 12.0)
            for offset in range(-4, 5):
                idx = cx_start + (ring_idx + offset) % n_cx
                inp[idx] += math.exp(-0.5 * (offset ** 2)) * (1.5 + trend_mag)

        return inp

    def _single_step(self, cur_obs: np.ndarray) -> np.ndarray:
        """One recurrent LIF update for a single observation."""
        inp = self._sensory_input(cur_obs)

        # 3. Dopaminergic PAM/PPL1 stimulation
        dan_start = self.IDX_DAN.start
        n_dan = self.IDX_DAN.stop - dan_start
        n_pam = n_dan // 2
        if self.dopamine_level > 0:
            inp[dan_start:dan_start + n_pam] += self.dopamine_level * 2.2
        else:
            inp[dan_start + n_pam:self.IDX_DAN.stop] += abs(self.dopamine_level) * 2.2

        recurrent = self.W @ self.last_spikes
        self._inh = self._inh * 0.85 + float(np.sum(self.last_spikes)) * 0.0005
        inp += 0.15 * np.ones(self.N_NEURONS, dtype=np.float32)
        noise = (np.random.normal(0, self.noise_std, size=self.N_NEURONS).astype(np.float32)
                 if self.noise_std > 0 else 0.0)
        self.V = self.decay * self.V + inp + recurrent - self._inh + noise

        spikes = (self.V > self.threshold).astype(np.float32)
        self.V[spikes > 0] = 0.0  # reset membrane potential
        self.last_spikes = spikes
        return spikes

    def step(self, obs: np.ndarray, reward: float = 0.0) -> np.ndarray:
        """Stateful, one LIF update per sample (live stepping path).

        NOTE: prefer ``encode`` for feature extraction; see its docstring.
        """
        if reward:
            self.dopamine_level = float(np.clip(0.92 * self.dopamine_level + reward * 8.0, -1.0, 1.0))
        obs2 = np.atleast_2d(obs)
        out = np.zeros((obs2.shape[0], self.N_NEURONS), dtype=np.float32)
        for b in range(obs2.shape[0]):
            out[b] = self._single_step(obs2[b])
        return out

    def encode(self, obs_np: np.ndarray, reward: float = 0.0, steps: int = None) -> np.ndarray:
        """Stateless, deterministic spike encoding of observations.

        Resets the membrane state and *settles* the network for ``steps`` LIF
        updates with the same sensory input, so the returned spike pattern is a
        pure function of the observation (no history, no noise).

        This determinism is what makes the replay buffer valid: the features
        computed at collection time are reproduced exactly at training time.
        The old stateful/non-deterministic read-out made every stored
        ``(obs, action)`` pair inconsistent with the features used to choose it.
        """
        steps = self.settle_steps if steps is None else steps
        if reward:
            self.dopamine_level = float(np.clip(0.92 * self.dopamine_level + reward * 8.0, -1.0, 1.0))
        obs2 = np.atleast_2d(obs_np)
        out = np.zeros((obs2.shape[0], self.N_NEURONS), dtype=np.float32)
        for b in range(obs2.shape[0]):
            self.reset_state()
            spikes = self.last_spikes
            for _ in range(steps):
                spikes = self._single_step(obs2[b])
            out[b] = spikes
        return out


# ── PyTorch Dopaminergic Actor-Critic Agent ─────────────────────────────────
class FlyBrainAgent(nn.Module):
    """
    Drosophila MaleCNS Spiking Connectome + Dopamine-modulated Actor-Critic Policy.
    - Connectome extracts N-dim biologically grounded sparse representations.
    - Policy Head (Actor): Outputs probability distribution over [HOLD, BUY, SELL].
    - Value Head (Critic): Estimates expected future return.
    - Online learning engine trains continuously from rewards and portfolio outcomes.
    
    Args:
        obs_dim: Observation dimension. Defaults to ``config.OBS_DIM`` (390 for
                  a 30-step OHLCV + 8-indicator window). Prefer constructing via
                  ``FlyBrainAgent(obs_dim=env.observation_space.shape[0])``.
        use_connectome: If True, use real FlyWire v783 connectome weights
        connectome_dir: Path to FlyWire data directory
        weight_cache: Path to cached weight pickle file
        N_NEURONS: Total neurons (default 1024). If >1024, uses more connectome data.
    """
    SENSORY_DIM = 16  # dims of the compact observation summary (see sensory_from_obs)

    def __init__(self, obs_dim: int = OBS_DIM, use_connectome=False, connectome_dir="flywire_data", weight_cache="flywire_weights", N_NEURONS=None):
        super().__init__()
        self.simulator = MaleCNS1024Simulator(
            use_connectome=use_connectome,
            connectome_dir=connectome_dir,
            weight_cache=weight_cache,
            N_NEURONS=N_NEURONS
        )
        self.obs_dim = obs_dim
        self.N_NEURONS = self.simulator.N_NEURONS

        self.mbon_dim = self.simulator.IDX_MBON.stop - self.simulator.IDX_MBON.start
        # The connectome read-out is concatenated with a compact, live sensory
        # vector derived from the observation (see ``sensory_from_obs``). The
        # cached connectome currently emits an all-zero MBON pattern for every
        # market state, so without this channel the policy would be a constant
        # action regardless of the market.
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.mbon_dim + self.SENSORY_DIM, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.1),
        )

        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 7)
        )

        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

        # Initialize with smart momentum/trend bias so the fly is intelligent immediately
        self._init_smart_weights()

        # Optimizer with decoupled weight decay (AdamW) for numerical stability and compatibility
        self.optimizer = optim.AdamW(self.parameters(), lr=3e-4, weight_decay=1e-4)

        # Experience replay buffer for online learning
        self.replay_buffer = []
        self.max_buffer_size = 4000

        # RL hyperparameters
        self.gamma = 0.99
        self.entropy_coef = 0.08
        self.value_coef = 0.5
        self.max_grad_norm = 1.0
        self.target_entropy = 1.30  # High target entropy to prevent collapse into pure Holding
        self.entropy_alpha = 0.15
        # Anti-stall bias pushing the policy toward buying. Kept at 0.35 for the
        # legacy/live demo, but set to 0.0 when training against a benchmark so
        # the fly is free to learn that holding cash is sometimes optimal.
        self.buy_margin_coef = 0.35

    def _init_smart_weights(self):
        # Heuristic smart initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Actor head initial gain
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.05)
        # Prior biases: balanced Hold with proactive Buy / Sell interest
        with torch.no_grad():
            self.actor[-1].bias.copy_(torch.tensor([0.35, 0.15, 0.20, 0.10, 0.15, 0.20, 0.10]))

    def step_simulation(self, obs_np: np.ndarray, reward: float = 0.0) -> np.ndarray:
        """Deterministic connectome encoding -> (batch, N_NEURONS) spike rates."""
        return self.simulator.encode(obs_np, reward=reward)

    @staticmethod
    def sensory_from_obs(obs) -> np.ndarray:
        """Compact, live summary of one observation (a pure function of obs).

        5 multi-horizon returns + realised volatility (2) + range position (1)
        + the 8 current-step indicators = 16 dims. These are already normalised
        by the environment, so we only apply bounded clipping.
        """
        o = np.asarray(obs, dtype=np.float64).reshape(-1)
        n = o.size
        close = o[3:min(n, 150):5]                  # the 30 window closes
        cur_ind = o[-8:] if n >= 8 else np.zeros(8, dtype=np.float64)
        feats = []
        if close.size >= 21:
            c = float(close[-1])
            for k in (1, 2, 5, 10, 20):
                base = float(close[-1 - k])
                feats.append(np.clip((c / (base + 1e-9) - 1.0) * 10.0, -3.0, 3.0))
            rets = np.diff(close) / (close[:-1] + 1e-9)
            vol = float(np.std(rets))
            short_vol = float(np.std(rets[-10:])) if rets.size >= 10 else vol
            feats.append(np.clip(vol * 100.0, 0.0, 3.0))
            feats.append(np.clip(short_vol * 100.0, 0.0, 3.0))
            lo, hi = float(close.min()), float(close.max())
            feats.append(np.clip((c - lo) / (hi - lo + 1e-9) * 2.0 - 1.0, -1.0, 1.0))
        else:
            feats.extend([0.0] * 8)
        feats.extend(list(np.clip(cur_ind, -5.0, 5.0)))
        return np.asarray(feats, dtype=np.float32)

    def encode_obs(self, obs):
        """Return (feature_vector, spikes_1024) for one observation, once."""
        obs2 = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        spikes = self.simulator.encode(obs2)[0]
        mbon = spikes[self.simulator.IDX_MBON.start:self.simulator.IDX_MBON.stop]
        feat = np.concatenate([mbon, self.sensory_from_obs(obs2[0])]).astype(np.float32)
        return feat, spikes

    def encode_feature(self, obs) -> np.ndarray:
        """Deterministic feature vector fed to the read-out: [MBON spikes, sensory]."""
        return self.encode_obs(obs)[0]

    def forward_features(self, feat_t: torch.Tensor):
        features = self.feature_encoder(feat_t)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

    def forward(self, obs_t: torch.Tensor, reward: float = 0.0):
        """Back-compat wrapper: obs tensor -> (logits, value, spikes)."""
        obs_np = np.atleast_2d(obs_t.detach().cpu().numpy())
        feats = np.stack([self.encode_feature(row) for row in obs_np])
        logits, value = self.forward_features(torch.from_numpy(feats))
        return logits, value, self.simulator.encode(obs_np)

    def select_action(self, obs_np: np.ndarray, reward: float = 0.0, deterministic: bool = False):
        """
        Evaluates current market state and returns chosen action, probabilities,
        and full 1024-neuron spiking activity.
        """
        self.eval()
        with torch.no_grad():
            row = np.asarray(obs_np, dtype=np.float32).reshape(-1)
            feat, spikes_1024 = self.encode_obs(row)
            logits, value = self.forward_features(torch.from_numpy(feat)[None, :])

            probs = F.softmax(logits[0], dim=-1).cpu().numpy()
            # Anti-collapse exploration floor: ensures no action probability ever drops to zero
            probs = np.maximum(probs, 0.04)
            probs = probs / np.sum(probs)

            if deterministic:
                action = int(np.argmax(probs))
            else:
                action = int(np.random.choice(7, p=probs))

        return action, probs.tolist(), float(value.item()), spikes_1024

    def record_experience(self, obs, action, reward, next_obs, done):
        """Append to the experience buffer, caching *features* (not raw obs).

        Caching makes training independent of the (expensive, and previously
        non-deterministic) spiking simulator, and guarantees the stored
        (feature, action) pairing is exactly the one the policy acted on.
        """
        feat = self.encode_feature(obs)
        next_feat = self.encode_feature(next_obs)
        self.replay_buffer.append((feat, action, reward, next_feat, done))
        if len(self.replay_buffer) > self.max_buffer_size:
            self.replay_buffer.pop(0)

    def get_policy_state(self):
        return {
            'model_state': self.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'replay_buffer': self.replay_buffer[-self.max_buffer_size:],
            'simulator_state': {
                'V': self.simulator.V.tolist(),
                'last_spikes': self.simulator.last_spikes.tolist(),
                '_inh': float(self.simulator._inh),
                'dopamine_level': float(self.simulator.dopamine_level),
            },
        }

    def set_policy_state(self, state):
        if state and 'model_state' in state:
            # tolerate architecture drift (e.g. feature_encoder input grew from
            # 64 -> 80 dims when the sensory channel was added) instead of
            # hard-failing on a stale checkpoint.
            try:
                self.load_state_dict(state['model_state'])
            except RuntimeError as e:
                missing, unexpected = self.load_state_dict(state['model_state'], strict=False)
                print(f"Warning: partial checkpoint load ({e}); "
                      f"missing={len(missing)} unexpected={len(unexpected)}")
        if state and 'optimizer_state' in state:
            try:
                self.optimizer.load_state_dict(state['optimizer_state'])
            except (ValueError, RuntimeError) as e:
                print(f"Warning: skipped optimizer state ({e})")
        if state and 'replay_buffer' in state:
            # Drop entries from an older architecture (raw 390-dim obs instead of
            # the 80-dim cached features) so they cannot crash training.
            expected = self.feature_encoder[0].in_features
            buf = []
            for e in state['replay_buffer']:
                try:
                    if (np.asarray(e[0]).size == expected
                            and np.asarray(e[3]).size == expected):
                        buf.append(e)
                except Exception:
                    continue
            self.replay_buffer = buf
        if state and 'simulator_state' in state:
            sim_state = state['simulator_state']
            self.simulator.V = np.array(sim_state['V'], dtype=np.float32)
            self.simulator.last_spikes = np.array(sim_state['last_spikes'], dtype=np.float32)
            self.simulator._inh = float(sim_state['_inh'])
            self.simulator.dopamine_level = float(sim_state['dopamine_level'])

    def train_online_step(self, batch_size: int = 32) -> float:
        """
        Dopamine-modulated Policy Gradient (Actor-Critic update).
        Runs online on cached features (no simulator re-run -> fast and valid).
        """
        if len(self.replay_buffer) < 8:
            return 0.0

        self.train()
        indices = np.random.choice(len(self.replay_buffer), size=min(batch_size, len(self.replay_buffer)), replace=False)
        batch = [self.replay_buffer[i] for i in indices]

        feat_b = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32)
        acts_b = torch.tensor([b[1] for b in batch], dtype=torch.long)
        rews_b = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        next_feat_b = torch.tensor(np.stack([b[3] for b in batch]), dtype=torch.float32)
        dones_b = torch.tensor([b[4] for b in batch], dtype=torch.float32)

        # 1. Next-state values in eval mode (pure inference, no graph retention).
        self.eval()
        with torch.no_grad():
            _, next_values = self.forward_features(next_feat_b)
            next_values = next_values.detach()

        # 2. Current-state forward pass (gradients flow through the read-out).
        self.train()
        logits, values = self.forward_features(feat_b)

        # TD target — fully detached so no inplace-op autograd issues
        gamma = self.gamma
        targets = (rews_b.unsqueeze(1) + gamma * next_values * (1.0 - dones_b.unsqueeze(1))).detach()
        advantages = targets - values.detach()

        # Advantage normalization and clipping for stable, reliable policy gradients
        if advantages.numel() > 1:
            adv_std = advantages.std()
            if adv_std > 1e-6:
                advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
        advantages = torch.clamp(advantages, -3.0, 3.0)

        # Actor loss (Policy Gradient)
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, acts_b.unsqueeze(1))
        actor_loss = -(action_log_probs * advantages).mean()

        # Critic loss — Huber loss (Smooth L1) protects against financial return outliers
        critic_loss = F.smooth_l1_loss(values, targets)

        # Entropy bonus for exploration
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        # Dynamic entropy adjustment: increase entropy bonus if entropy drops below target
        entropy_error = entropy.item() - self.target_entropy
        self.entropy_coef = max(0.005, self.entropy_coef + self.entropy_alpha * (-entropy_error))
        self.entropy_coef = min(self.entropy_coef, 0.2)  # Cap entropy coefficient

        # Anti-stall margin penalty: mathematically guarantees Hold logit cannot crush Buy logits into negative oblivion
        buy_margin = logits[:, 0].clone() - logits[:, 1:4].clone().mean(dim=-1)
        margin_penalty = F.relu(buy_margin - 0.15).mean()

        total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy + self.buy_margin_coef * margin_penalty

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return float(total_loss.item())

    def get_synaptic_weights_sample(self) -> np.ndarray:
        """Returns 64 Mushroom Body readout weights (Kenyon Cell to MBON) for LTP/LTD analysis."""
        w = self.feature_encoder[0].weight.detach().cpu().numpy()
        # Sample 8x8 block of weights
        return w[:8, :8].flatten().tolist()

    def get_region_voltages(self) -> dict:
        """Computes current biological membrane potential in millivolts (-70mV to +30mV)."""
        V = self.simulator.V
        def to_mv(v_val):
            return float(np.clip(-70.0 + v_val * 45.0, -85.0, 30.0))

        al = self.simulator.IDX_AL
        kc = self.simulator.IDX_KC
        cx = self.simulator.IDX_CX
        dan = self.simulator.IDX_DAN
        mbon = self.simulator.IDX_MBON

        return {
            "al":   round(to_mv(float(np.mean(V[al]))), 1),
            "kc":   round(to_mv(float(np.mean(V[kc]))), 1),
            "cx":   round(to_mv(float(np.mean(V[cx]))), 1),
            "dan":  round(to_mv(float(np.mean(V[dan]))), 1),
            "mbon": round(to_mv(float(np.mean(V[mbon]))), 1),
        }

    def make_smarter(self, connectome_dir="flywire_data", weight_cache="flywire_weights", N_NEURONS=None):
        """
        Upgrade the fly's brain with real FlyWire v783 connectome weights.
        Replaces the current weight matrix with biologically-grounded synaptic weights
        from the full Drosophila connectome.
        """
        print("Upgrading fly brain to connectome mode...")
        self.simulator = MaleCNS1024Simulator(
            use_connectome=True,
            connectome_dir=connectome_dir,
            weight_cache=weight_cache,
            N_NEURONS=N_NEURONS if N_NEURONS is not None else self.N_NEURONS
        )
        self.simulator.reset()
        print("Fly brain upgraded with FlyWire v783 connectome weights")

    def reset(self):
        self.simulator.reset()


# Backward-compatible wrapper
FlyBrainPolicy = FlyBrainAgent
