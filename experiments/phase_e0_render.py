"""Layout-only repair of E0 contact sheets. No changes to data or case selection."""
import csv
import sys
import numpy as np
from PIL import Image
from experiments import phase_e0 as e

def render():
    sys.meta_path.insert(0,e.NoModels())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    out=e.ROOT/e.OUT
    with (out/'persistent_failure_matrix.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    ordered=sorted(rows,key=lambda r:(e.CATEGORIES.index(r['category']),r['size_bin'],r['sample_id'],int(r['GT_instance_id'])))
    manifest={r['sample_id']:r for r in e.read(e.ROOT/'experiments/protocols/ISIC_FUSEG_COLORFIX_V1_validation_manifest.json')['samples']}
    for start in range(0,len(ordered),9):
        fig,axes=plt.subplots(3,3,figsize=(12,16))
        for ax in axes.flat:ax.axis('off')
        for ax,r in zip(axes.flat,ordered[start:start+9]):
            sample=manifest[r['sample_id']]
            p=e.assert_relative_admitted(e.ROOT,sample['relative_image_path'],e.ROOT/'outputs/fuseg_warmup_revision_20260914/dataset/images/val')
            e.require(e.d2.sha(p)==sample['image_sha256'],'render input hash')
            with Image.open(p) as im:rgb=np.array(im.convert('RGB'))
            ax.imshow(rgb);b=[float(r[k]) for k in ('bbox_x1','bbox_y1','bbox_x2','bbox_y2')]
            ax.add_patch(Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,edgecolor='#ffff00',linewidth=2))
            ax.set_title(f"{r['sample_id']} G{r['GT_instance_id']} {r['size_bin']}\n{r['category']}\nC={r['control_detection_count']}/5 S={r['experimental_detection_count']}/5",fontsize=8,pad=10)
        fig.suptitle(f'All very-small GT, page {start//9+1}/6\nRule-sorted; yellow=frozen target GT; display only',fontsize=12,y=.975)
        fig.subplots_adjust(left=.025,right=.975,bottom=.02,top=.92,wspace=.12,hspace=.32)
        fig.savefig(out/'failure_visualizations'/f'all_instances_{start//9+1:02d}.png',dpi=120);plt.close(fig)
    print('Six pages re-laid out; same 49 GT, same source pixels and boxes.')

if __name__=='__main__':render()
