#!/usr/bin/env python3
"""Plot completed human-only BC training and paired deployment predictions."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/pipette-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[2]/'outputs/pi05-hil293-human-bc-20260914-native')
    args=p.parse_args();out=args.output
    rows=[json.loads(x) for x in (out/'metrics.jsonl').read_text().splitlines()]
    train=[x for x in rows if 'train_loss' in x];val=[x for x in rows if 'validation_loss' in x]
    fig,ax=plt.subplots(figsize=(9,4.6),layout='constrained')
    tx=np.array([x['step'] for x in train]);ty=np.array([x['train_loss'] for x in train])
    ax.plot(tx,ty,color='#afbecb',alpha=.7,lw=1,label='Training minibatches (logged every 10 updates)')
    if len(ty)>=5:ax.plot(tx[4:],np.convolve(ty,np.ones(5)/5,mode='valid'),color='#2878a8',label='Training: 5-point moving average')
    vx=[x['step'] for x in val];vy=[x['validation_loss'] for x in val]
    ax.plot(vx,vy,'o-',color='#db6b25',lw=2,label='Held-out episodes, fixed flow noise')
    for x,y in zip(vx,vy):ax.annotate(f'{y:.3f}',(x,y),xytext=(4,9),textcoords='offset points')
    ax.set(xlabel='Optimizer update',ylabel='Normalized native flow MSE',title='HIL293 10-epoch BC → human corrections only')
    ax.set_ylim(bottom=0);ax.grid(alpha=.2);ax.legend(fontsize=9)
    for ext in ('png','pdf'):fig.savefig(out/f'loss_curve.{ext}',dpi=170)
    plt.close(fig)
    paths=[out/(name+'-evaluation.json') for name in ('baseline','trained')]
    if all(p.exists() for p in paths):
        old,new=[json.loads(p.read_text()) for p in paths]
        fig,ax=plt.subplots(figsize=(7,4.5),layout='constrained');x=np.arange(3)
        for offset,result,label,color in [(-.18,old,'Original BC','#748fa5'),(.18,new,'Human-only BC','#dc7c39')]:
            bars=ax.bar(x+offset,result['rmse_mm'],.36,label=label,color=color)
            ax.bar_label(bars,fmt='%.3f',padding=3)
        ax.set(xticks=x,xticklabels=['X','Y','Z'],ylabel='Action RMSE (mm per 30 Hz step)',
               title=f"Held-out human corrections: {old['human_targets']} targets / {old['observations']} observations")
        ax.legend();ax.set_ylim(0,max(old['rmse_mm']+new['rmse_mm'])*1.2);ax.grid(axis='y',alpha=.2)
        for ext in ('png','pdf'):fig.savefig(out/f'action_rmse.{ext}',dpi=170)
        plt.close(fig)
    print(out/'loss_curve.png')


if __name__=='__main__':main()
