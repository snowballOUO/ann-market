"""Rebuild main_results.csv from console outputs."""
import pandas as pd

data = []

def add(ds, seed, **methods):
    for m, (rev, acc) in methods.items():
        data.append({"dataset": ds, "method": m, "seed": seed,
                      "revenue": rev, "accept_rate": acc})

# ── AG_NEWS (latest run with skip-train) ──
ag = [
    (42, {"fixed":(49.3586,0.988),"sla":(48.3427,0.999),"cost":(48.5039,0.999),
          "linucb":(58.4262,0.911),"naive_dqn":(69.8155,0.870),"qnet":(67.0152,0.880)}),
    (123,{"fixed":(49.1426,0.989),"sla":(48.3054,0.999),"cost":(48.4547,0.999),
          "linucb":(57.3521,0.905),"naive_dqn":(70.1893,0.870),"qnet":(67.4231,0.876)}),
    (456,{"fixed":(49.0534,0.989),"sla":(48.3114,0.999),"cost":(48.4796,0.999),
          "linucb":(58.0634,0.911),"naive_dqn":(69.1783,0.870),"qnet":(67.5078,0.871)}),
    (789,{"fixed":(49.3515,0.991),"sla":(48.3641,0.999),"cost":(48.5252,0.999),
          "linucb":(57.6281,0.910),"naive_dqn":(69.8917,0.874),"qnet":(66.9217,0.872)}),
    (101112,{"fixed":(49.4367,0.990),"sla":(48.3397,0.999),"cost":(48.5032,0.999),
             "linucb":(58.3559,0.915),"naive_dqn":(70.1706,0.869),"qnet":(68.2644,0.878)}),
]
for s, m in ag:
    add("ag_news", s, **m)

# ── SIFT1M (latest run with skip-train) ──
sift = [
    (42, {"fixed":(47.3693,0.950),"sla":(46.1824,0.961),"cost":(46.7387,0.963),
          "linucb":(52.4281,0.855),"naive_dqn":(51.4688,0.864),"qnet":(51.7251,0.862)}),
    (123,{"fixed":(47.0607,0.951),"sla":(46.1486,0.961),"cost":(46.6833,0.962),
          "linucb":(51.6486,0.852),"naive_dqn":(50.5790,0.857),"qnet":(50.0280,0.861)}),
    (456,{"fixed":(46.9153,0.952),"sla":(46.1517,0.961),"cost":(46.7382,0.963),
          "linucb":(52.2191,0.857),"naive_dqn":(50.3990,0.865),"qnet":(52.0903,0.865)}),
    (789,{"fixed":(46.8487,0.950),"sla":(46.1243,0.961),"cost":(46.6682,0.962),
          "linucb":(51.6061,0.855),"naive_dqn":(50.5139,0.862),"qnet":(51.2558,0.860)}),
    (101112,{"fixed":(47.3861,0.955),"sla":(46.2494,0.963),"cost":(46.7452,0.964),
             "linucb":(52.4432,0.862),"naive_dqn":(50.1109,0.869),"qnet":(52.6557,0.863)}),
]
for s, m in sift:
    add("sift1m", s, **m)

# ── DEEP1M (from first run) ──
deep = [
    (42, {"fixed":(43.5841,0.884),"sla":(42.8569,0.897),"cost":(43.4540,0.898),
          "linucb":(45.2772,0.778),"naive_dqn":(37.3025,0.812),"qnet":(54.1651,0.717)}),
    (123,{"fixed":(43.5187,0.887),"sla":(42.7757,0.896),"cost":(43.4491,0.898),
          "linucb":(44.7268,0.775),"naive_dqn":(35.8017,0.811),"qnet":(54.3724,0.703)}),
    (456,{"fixed":(43.3336,0.886),"sla":(42.8355,0.897),"cost":(43.4427,0.898),
          "linucb":(45.1726,0.783),"naive_dqn":(36.7787,0.812),"qnet":(53.5922,0.710)}),
    (789,{"fixed":(43.4710,0.885),"sla":(42.7400,0.895),"cost":(43.4305,0.897),
          "linucb":(44.9311,0.779),"naive_dqn":(37.1106,0.805),"qnet":(53.6559,0.698)}),
    (101112,{"fixed":(43.6594,0.890),"sla":(42.9927,0.900),"cost":(43.6090,0.901),
             "linucb":(45.6418,0.789),"naive_dqn":(37.2740,0.809),"qnet":(52.1411,0.715)}),
]
for s, m in deep:
    add("deep1m", s, **m)

# ── GIST1M (from console) ──
gist = [
    (42, {"fixed":(47.5906,0.985),"sla":(42.7781,0.990),"cost":(46.4225,0.996),
          "linucb":(55.3472,0.905),"naive_dqn":(56.0273,0.900),"qnet":(69.4748,0.875)}),
    (123,{"fixed":(46.7803,0.985),"sla":(44.2818,0.993),"cost":(46.6685,0.996),
          "linucb":(54.9222,0.902),"naive_dqn":(54.0909,0.896),"qnet":(69.7732,0.870)}),
    (456,{"fixed":(47.2503,0.986),"sla":(44.1008,0.992),"cost":(46.6542,0.995),
          "linucb":(55.4053,0.905),"naive_dqn":(50.3539,0.896),"qnet":(70.4026,0.866)}),
    (789,{"fixed":(46.7981,0.985),"sla":(44.0156,0.990),"cost":(46.8401,0.996),
          "linucb":(54.9766,0.905),"naive_dqn":(50.6696,0.903),"qnet":(70.4862,0.868)}),
    (101112,{"fixed":(47.5007,0.987),"sla":(44.0221,0.991),"cost":(46.7374,0.996),
             "linucb":(55.6229,0.910),"naive_dqn":(55.0545,0.899),"qnet":(69.6099,0.869)}),
]
for s, m in gist:
    add("gist1m", s, **m)

# ── Save ──
df = pd.DataFrame(data)
df.to_csv("reports/main_results.csv", index=False)
print(f"Saved {len(df)} rows across {df['dataset'].nunique()} datasets")
print()
summary = df.groupby(["dataset", "method"]).agg(
    revenue_mean=("revenue", "mean"),
    revenue_std=("revenue", "std"),
    accept_mean=("accept_rate", "mean"),
    n=("revenue", "count"),
).round(4)
print(summary.to_string())
