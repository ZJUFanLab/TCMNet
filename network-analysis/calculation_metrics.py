import networkx as nx
import random
import numpy as np
from scipy.stats import spearmanr
from network_utilities import get_degree_binning, pick_random_nodes_matching_selected

# ─── 缓存结构 ─────────────────────────────
_distance_cache = {}

def weighted_shortest_path(G, src, tgt, w_src, w_tgt):
    vals = []
    for s in src:
        if s not in G: continue
        if s not in _distance_cache:
            _distance_cache[s] = nx.single_source_shortest_path_length(G, s)
        dist_s = _distance_cache[s]
        ws = w_src.get(s, 1.0)
        best = np.inf
        for t in tgt:
            if s == t: continue  
            d = dist_s.get(t)
            if d is None: continue
            wt = w_tgt.get(t, 1.0)
            if ws == 0 or wt == 0: continue
            best = min(best, d/(ws*wt))
        if best < np.inf:
            vals.append(best)
    return np.mean(vals) if vals else np.nan

def calculate_separation(G, drug, disease, drug_w, disease_w):

    dHD = proximity_weighted(G, drug, disease, drug_w, disease_w)  

    fake_drug_w = {k: 1.0 for k in drug}
    dHH = proximity_weighted(G, drug, drug, fake_drug_w, fake_drug_w)  

    fake_dis_w = {k: 1.0 for k in disease}
    dDD = proximity_weighted(G, disease, disease, fake_dis_w, fake_dis_w)  

    return dHD - 0.5 * (dHH + dDD)  

def get_random_nodes(network, nodes, n_random=100, min_bin_size=100,
                     degree_aware=True, seed=None):
    bins = get_degree_binning(network, min_bin_size)
    return pick_random_nodes_matching_selected(
        network, bins, nodes, n_random, degree_aware, seed=seed)

def separation_zscore(G, drug, disease, drug_w, disease_w,
                     iterations=100, sampler=None):
    """
    Z-score for Separation: compare actual separation to random drug/disease sets.
    """
    sep_obs = calculate_separation(G, drug, disease, drug_w, disease_w)
    if np.isnan(sep_obs):
        return np.nan
    if sampler is None:
        def sampler(dr, ds):
            all_nodes = list(G.nodes())
            return (random.sample(all_nodes, len(dr)),
                    random.sample(all_nodes, len(ds)))
    scores = []
    for _ in range(iterations):
        rd, rD = sampler(drug, disease)
        val = calculate_separation(G, rd, rD, drug_w, disease_w)
        if not np.isnan(val):
            scores.append(val)
    if not scores:
        return np.nan
    mu, sd = np.mean(scores), np.std(scores)
    return (sep_obs - mu)/sd if sd else np.nan

def proximity_weighted(G, A, B, wA, wB):
    # 新增：计算所有 A->B 的最短路径，乘以 ws*wt，再归一化
    total_weighted_distance = 0.0  
    total_weight = 0.0  

    for a in A:
        if a not in G:
            continue
        if a not in _distance_cache:
            _distance_cache[a] = nx.single_source_shortest_path_length(G, a)
        dist_a = _distance_cache[a]
        ws = wA.get(a, 1.0)
        for b in B:
            if b == a:  
                continue
            d = dist_a.get(b)
            if d is None:
                continue
            wt = wB.get(b, 1.0)
            weight = ws * wt
            total_weighted_distance += d * weight
            total_weight += weight

    
    return total_weighted_distance / total_weight if total_weight > 0 else np.nan

def proximity_weighted_zscore(G, drug, disease, drug_w, disease_w,
                              iterations=100, sampler=None):
    """
    Z-score for proximity_weighted: compare actual weighted proximity
    to random sets.
    """
    prox_obs = proximity_weighted(G, drug, disease, drug_w, disease_w)
    if np.isnan(prox_obs):
        return np.nan
    if sampler is None:
        def sampler(dr, ds):
            all_nodes = list(G.nodes())
            return (random.sample(all_nodes, len(dr)),
                    random.sample(all_nodes, len(ds)))
    scores = []
    for _ in range(iterations):
        rd, rD = sampler(drug, disease)
        val = proximity_weighted(G, rd, rD, drug_w, disease_w)
        if not np.isnan(val):
            scores.append(val)
    if not scores:
        return np.nan
    mu, sd = np.mean(scores), np.std(scores)
    return (prox_obs - mu)/sd if sd else np.nan

def jaccard_index(A, B):
    A, B = set(A), set(B)
    if not A and not B: return 1.0
    u = len(A|B)
    return len(A&B)/u if u else 0.0

def coverage_overlap(A, B):
    A, B = set(A), set(B)
    return len(A&B)/len(A) if A else 0.0

def coverage_directlink(G, A, B):
    A, B = set(A), set(B)
    if not A: return 0.0
    nbrs = set()
    for b in B:
        if b in G:
            nbrs |= set(G.neighbors(b))
    covered = A & (B|nbrs)
    return len(covered)/len(A)

def robustness_node_removal(G, drug, disease, drug_w, disease_w,
                            fraction=0.1, iterations=50):
    base = calculate_separation(G, drug, disease, drug_w, disease_w)
    if np.isnan(base): return np.nan, np.nan
    cand = [n for n in G if n not in drug and n not in disease]
    remove_n = max(1, int(fraction*len(cand)))
    ratios = []
    for _ in range(iterations):
        rem = random.sample(cand, remove_n)
        H = G.copy()
        H.remove_nodes_from(rem)
        s = calculate_separation(H, drug, disease, drug_w, disease_w)
        if not np.isnan(s):
            ratios.append(s/base)
    return (np.mean(ratios), np.std(ratios)) if ratios else (np.nan, np.nan)

def robustness_weight_perturb(G, drug, disease, drug_w, disease_w,
                             perturb=0.1, iterations=50):
    base = calculate_separation(G, drug, disease, drug_w, disease_w)
    if np.isnan(base): return np.nan, np.nan
    ratios = []
    for _ in range(iterations):
        pw = {k:drug_w[k]*(1+random.uniform(-perturb,perturb))
              for k in drug_w}
        s = calculate_separation(G, drug, disease, pw, disease_w)
        if not np.isnan(s):
            ratios.append(s/base)
    return (np.mean(ratios), np.std(ratios)) if ratios else (np.nan, np.nan)

def spearman_weight_sensitivity(weight_map, steps=10):
    genes = list(weight_map.keys())
    orig = sorted(genes, key=lambda g: weight_map[g])
    cors = []
    for alpha in np.linspace(0,1,steps):
        pw = {g: weight_map[g]*(1-alpha) + alpha*random.random()
              for g in genes}
        newr = sorted(genes, key=lambda g: pw[g])
        coef,_ = spearmanr(
            [orig.index(g) for g in genes],
            [newr.index(g) for g in genes]
        )
        cors.append(coef)
    return cors

