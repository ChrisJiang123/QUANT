"""Standalone research figures from engine-derived series; no second backtester."""
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/v4'

def main():
    folder=ROOT/'reports/figures';folder.mkdir(exist_ok=True)
    fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True)
    models={'buy_hold':'Buy and hold','equal_exposure_hold':'Rule exposure-matched hold','rule':'Rule ensemble',
            'ridge':'Ridge','lightgbm':'LightGBM','sidgtn_seed11':'SIDGTN fixed seed'}
    sources=[]
    for axis,cost in zip(axes,['OPTIMISTIC','BASE','STRESS']):
        for color_id,(name,label) in enumerate(models.items()):
            path=OUT/f'tenyear_{name}_{cost}.parquet'
            if not path.exists():continue
            f=pd.read_parquet(path);sources.append(str(path.relative_to(ROOT)))
            for i,(segment,g) in enumerate(f.groupby('segment',sort=False)):
                axis.plot(g.date,g.research_index,label=label if i==0 else None,lw=1.2,color=plt.get_cmap('tab10')(color_id))
        axis.set_yscale('log');axis.set_ylabel(cost+'\nResearch index');axis.grid(alpha=.2)
        axis.axvspan(pd.Timestamp('2025-01-29',tz='UTC'),pd.Timestamp('2026-09-04',tz='UTC'),color='orange',alpha=.08)
        axis.axvspan(pd.Timestamp('2026-10-02',tz='UTC'),pd.Timestamp('2026-11-13',tz='UTC'),color='green',alpha=.08)
    axes[0].legend(ncol=3,fontsize=8,loc='upper left')
    fig.suptitle('Costed OOS research index — distinct DEV / touched / fresh evidence segments\nSeparate account resets; gaps omitted; fresh anchors insufficient; no Alpha promotion',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(folder/'V4_COSTED_RESEARCH_INDEX.png',dpi=160);fig.savefig(folder/'V4_COSTED_RESEARCH_INDEX.svg');plt.close(fig)
    target=OUT/'figures';target.mkdir(exist_ok=True);(target/'metrics.json').write_text(json.dumps(dict(schema_version=1,sources=sorted(set(sources)),
             outputs=['reports/figures/V4_COSTED_RESEARCH_INDEX.png','reports/figures/V4_COSTED_RESEARCH_INDEX.svg'],performance_source='Freqtrade wallet exports'),indent=2))
    print('COSTED_RESEARCH_FIGURES_RENDERED')
if __name__=='__main__':main()
