"""Compute expected revenue for each (nprobe, price) under current buyer mix."""
import numpy as np, math

buyers = [
    ('Budget', 800, 20, 5,   0.005, 0.200, 0.70, 0.4),
    ('Latency', 50, 3000, 30, 0.050, 0.001, 0.90, 0.3),
    ('Quality', 200, 50, 60, 0.015, 0.100, 0.92, 0.3),
]
nprobes = [(8,0.66,0.00056),(16,0.79,0.00090),(32,0.88,0.00154),(64,0.93,0.00277),(128,0.95,0.00525)]
prices = [0.001, 0.002, 0.005, 0.01, 0.02]
bias = 0.6

results = []
for np_, recall, lat in nprobes:
    cost = 0.0001 + 0.00005 * lat
    for p in prices:
        ev = 0.0
        for name, ap, al, ar, tp, tl, tr, w in buyers:
            u = bias + ap*(tp-p) + al*(tl-lat) + ar*(recall-tr)
            u = max(-20, min(20, u))
            accept = 1.0/(1.0+math.exp(-u))
            ev += w * (accept * p - cost)
        results.append((ev, np_, p, ev*10000))
        print(f"  np{np_:3d} p=${p:.3f}  ev/10K=${ev*10000:6.2f}")
    print()

best = max(results, key=lambda x: x[0])
print(f"Optimal: nprobe={best[1]}  p=${best[2]:.3f}  ev/10K=${best[3]:.2f}")
