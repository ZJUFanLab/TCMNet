import os
import csv
import argparse
import networkx as nx
from calculation_metrics import *

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-f','--formula',       required=True,
                   help="Formula CSV with [protein, weight]")
    p.add_argument('-d','--disease_dir',   required=True,
                   help="Directory of topXX.csv disease files")
    p.add_argument('-n','--network',       required=True,
                   help="PPI edge‐list CSV, columns head,tail[,relation]")
    p.add_argument('-m','--metrics',       nargs='*', default=None,
                   help="Which metrics to run")
    p.add_argument('-i','--iterations',    type=int, default=100,
                   help="Randomization rounds for Z-scores/robustness")
    p.add_argument('-o','--out',           default='metric-results.csv',
                   help="Output CSV")
    p.add_argument('--spearman_out', default='spearman-weight-sensitivity.csv',
                   help="Output for Spearman weight sensitivity") 
    return p.parse_args()

def load_weights(path):
    wm = {}
    with open(path) as f:
        rdr = csv.reader(f)
        next(rdr)  # skip header
        for r in rdr:
            try:
                wm[r[0]] = float(r[1])
            except:
                continue
    return wm

def main():
    args = parse_args()

    # Available metrics and their callables
    allm = {
        'separation':            calculate_separation,
        'separation_zscore':     separation_zscore,
        'proximity':             proximity_weighted,
        'proximity_zscore':      proximity_weighted_zscore,
        'jaccard':               jaccard_index,
        'coverage_overlap':      coverage_overlap,
        'coverage_directlink':   coverage_directlink,
        'robustness_node':       robustness_node_removal,
        'robustness_weight':     robustness_weight_perturb,
        'spearman_weight_sensitivity': spearman_weight_sensitivity 
    }

    # If none specified, run all
    mets = args.metrics or list(allm.keys())

    # Load PPI network (ignore extra columns, take only first two)
    G = nx.Graph()
    with open(args.network) as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            u, v = row[0], row[1]
            G.add_edge(u, v)
    all_nodes = set(G.nodes())

    # Load formula node weights
    fw = load_weights(args.formula)
    drug_nodes = [n for n in fw if n in all_nodes]

    # Prepare output CSV
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='') as outf:
        wr = csv.writer(outf)
        wr.writerow(['top_file','scenario','metric','value'])

        # Spearman
        with open(args.spearman_out, 'w', newline='') as spearf:
            spearwr = csv.writer(spearf)
            spearwr.writerow(['top_file', 'scenario', 'weight_type', 'alpha', 'spearman_corr'])

            # Iterate over every topXX.csv in disease_dir
            for fn in sorted(os.listdir(args.disease_dir)):
                if not fn.startswith('top') or not fn.endswith('.csv'):
                    continue
                top_path = os.path.join(args.disease_dir, fn)

                # Load disease nodes & weights
                dw = {}
                dis_nodes = []
                with open(top_path) as f:
                    rdr = csv.reader(f)
                    next(rdr)  # skip header
                    for r in rdr:
                        node = r[0]
                        try:
                            wt = float(r[-1])
                        except:
                            wt = 1.0
                        if node in all_nodes:
                            dw[node] = wt
                            dis_nodes.append(node)

                # Define the four weight scenarios
                scen_cfg = {
                    'both_weights':  (fw,   dw),
                    'disease_only':  ({k:1.0 for k in fw}, dw),
                    'drug_only':     (fw,   {k:1.0 for k in dw}),
                    'no_weight':     ({k:1.0 for k in fw}, {k:1.0 for k in dw})
                }

                # For each scenario & each metric, compute and write a row
                for sc, (wA, wB) in scen_cfg.items():
                    for m in mets:
                        func = allm.get(m)
                        if func is None:
                            raise ValueError(f"Unknown metric '{m}'")

                        # Call with the correct signature
                        if m == 'separation':
                            val = func(G, drug_nodes, dis_nodes, wA, wB)
                        elif m == 'separation_zscore':
                            val = func(G, drug_nodes, dis_nodes, wA, wB,
                                       iterations=args.iterations)
                        elif m == 'proximity':
                            val = func(G, drug_nodes, dis_nodes, wA, wB)
                        elif m == 'proximity_zscore':
                            val = func(G, drug_nodes, dis_nodes, wA, wB,
                                       iterations=args.iterations)
                        elif m == 'jaccard':
                            val = func(dis_nodes, drug_nodes)
                        elif m == 'coverage_overlap':
                            val = func(dis_nodes, drug_nodes)
                        elif m == 'coverage_directlink':
                            val = func(G, dis_nodes, drug_nodes)
                        elif m == 'robustness_node':
                            val, _ = func(G, drug_nodes, dis_nodes, wA, wB,
                                          fraction=0.1,
                                          iterations=args.iterations)
                        elif m == 'robustness_weight':
                            val, _ = func(G, drug_nodes, dis_nodes, wA, wB,
                                          perturb=0.1,
                                          iterations=args.iterations)
                        elif m == 'spearman_weight_sensitivity': 
                            # 针对每种权重分别做一次敏感性分析并单独写入文件
                            for typ, wm in [('drug', wA), ('disease', wB)]: #药材/疾病
                                corrs = func(wm)
                                for idx, corr in enumerate(corrs):
                                    spearwr.writerow([fn, sc, typ, idx / (len(corrs)-1 if len(corrs)>1 else 1), corr])
                            val = None 
                        else:
                            val = None

                        if val is not None:
                            wr.writerow([fn, sc, m, val])

    print("✅ Done, results saved to", args.out)
    print("✅ Spearman weight sensitivity saved to", args.spearman_out) 

if __name__ == '__main__':
    main() 
