from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _joint_mask(obs=33, actions=126, block=21):
    mask=np.zeros((obs+actions,obs+actions),dtype=bool); mask[:obs,:obs]=True
    for q in range(actions): mask[obs+q,:obs+min((q//block+1)*block,actions)]=True
    return mask


def build(root: Path, out: Path) -> dict:
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/"prepared"/"action_manifest.json").read_text()); representation=manifest["config"]["data"].get("action_representation","bspline")
    action_steps=manifest["config"]["data"]["action_horizon"] if representation=="raw" else manifest["config"]["spline"]["num_basis"]
    action_tokens=action_steps*7; block=manifest["config"]["policy"]["block_size"]
    mask=_joint_mask(actions=action_tokens,block=block); fig,ax=plt.subplots(figsize=(7,7)); ax.imshow(mask,cmap="Blues",origin="upper",aspect="equal")
    for x in range(33,33+action_tokens+1,block): ax.axhline(x-.5,color="gray",lw=.5); ax.axvline(x-.5,color="gray",lw=.5)
    ax.set(xlabel="key position",ylabel="query position",title="Joint block-causal visibility (blue = visible)"); fig.tight_layout(); fig.savefig(out/"m3_joint_attention_mask.png",dpi=170); plt.close(fig)

    delay_files=sorted((root/"evaluation").glob("*_delays.json")); fig,ax=plt.subplots(figsize=(10,6)); delay_summary={}
    for path in delay_files:
        data=json.loads(path.read_text()); label=path.stem.removesuffix("_delays"); x=[c["raw_delay_actions"] for c in data["curves"]]; y=[c["decoded_physical_error"]["mae"] for c in data["curves"]]
        ax.plot(x,y,marker="o",label=label); delay_summary[label]={"d0_mae":y[0],"d10_mae":y[-1],"max_committed_error":max((c["committed_interval_preservation_normalized"] or {"max_abs":0})["max_abs"] for c in data["curves"])}
    if representation=="bspline":
        for x in (1,3,5,7,9): ax.axvspan(x-.12,x+.12,color="tab:orange",alpha=.08)
    xlabel="raw delay d (30 Hz actions; all d=1..10 seen in training)" if representation=="raw" else "raw delay d=s (30 Hz actions; orange bands are unseen within-span delays)"
    ax.set(xlabel=xlabel,ylabel="open-loop physical MAE",title="RTC delay curves on identical held-out samples",xticks=range(11)); ax.legend(fontsize=8,ncol=2); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(out/"m5_delay_curves.png",dpi=170); plt.close(fig)

    latency_files=sorted((root/"latency").glob("*.json")); fig,ax=plt.subplots(figsize=(10,6)); latency_summary={}
    for path in latency_files:
        data=json.loads(path.read_text()); label=path.stem; points=[]
        for key,value in data.items():
            if key.startswith("sampling_"): points.append((data["network_calls"][key],value["p50_ms"],key))
        for cache in ([False,True] if label.startswith("discrete_joint") else [None]):
            selected=[p for p in points if cache is None or f"use_cache-{cache}" in p[2]]
            selected.sort(); suffix="" if cache is None else (" cached" if cache else " full")
            if selected: ax.plot([p[0] for p in selected],[p[1] for p in selected],marker="o",label=label+suffix)
        latency_summary[label]={"online_vision_p50_ms":data["online_vision_total"]["p50_ms"],"sampling":{k:v for k,v in data.items() if k.startswith("sampling_")},"peak_memory_bytes":data["peak_memory_bytes"]}
    ax.set(xlabel="actual generative-network calls",ylabel="sampling p50 (ms)",title="Measured batch-one sampling latency"); ax.legend(fontsize=7,ncol=2); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(out/"m6_latency_curves.png",dpi=170); plt.close(fig)

    total=33+action_tokens; active_start=33+2*block; active_end=active_start+block
    fig,ax=plt.subplots(figsize=(11,2.8)); ax.set_xlim(0,total); ax.set_ylim(0,1); ax.axis("off")
    regions=[(0,33,"observation\ncacheable","tab:blue"),(33,active_start,"completed blocks\ncacheable","tab:green"),(active_start,active_end,"active block\nrecompute","tab:orange"),(active_end,total,"future blocks\nnot visible","tab:gray")]
    for start,end,text,color in regions: ax.add_patch(plt.Rectangle((start,.25),end-start,.45,color=color,alpha=.7)); ax.text((start+end)/2,.475,text,ha="center",va="center",fontsize=9)
    ax.text(total/2,.05,"Cache is plan-local; observation/state/model change or episode reset invalidates it.",ha="center",fontsize=10); ax.set_title("Exact joint-cache lifecycle under block-causal attention"); fig.tight_layout(); fig.savefig(out/"m6_cache_lifecycle.png",dpi=170); plt.close(fig)

    replay_files=sorted((root/"replay").glob("*_D3.json"))+sorted((root/"replay").glob("*_d3.json"))
    if replay_files:
        replay=json.loads(replay_files[0].read_text()); actions=np.array([x if x is not None else [np.nan]*7 for x in replay["executed"]]); fig,axes=plt.subplots(4,2,figsize=(12,10)); switches=[e["time"] for e in replay["timeline"] if e["event"]=="plan_switch"]
        t=np.arange(len(actions))/30
        for d,ax in enumerate(axes.flat[:7]): ax.plot(t,actions[:,d]); [ax.axvline(s,color="tab:red",alpha=.25) for s in switches]; ax.set_title(f"executed action {d}")
        axes.flat[7].axis("off"); delay_label="d=3 raw actions" if replay.get("action_representation")=="raw" else "D=S=3 spans"
        fig.suptitle(f"Dataset RTC replay, {replay['architecture']}, {delay_label}; red = plan switch"); fig.tight_layout(); fig.savefig(out/"m6_replay_switches.png",dpi=170); plt.close(fig)
    summary={"action_representation":representation,"delay":delay_summary,"latency":latency_summary,"scope":"measured open-loop/replay evidence; not closed-loop success"}; (out/"final_metrics_summary.json").write_text(json.dumps(summary,indent=2)+"\n"); return summary


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--root",default="outputs"); p.add_argument("--output",default="outputs/visualizations/final"); a=p.parse_args(argv); print(json.dumps(build(Path(a.root),Path(a.output)),indent=2))


if __name__ == "__main__": main()
