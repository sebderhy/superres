from fastai2.vision.all import * 
from utils import *

# def pltFigureOut(pred):
#     img_pil2=PILImage(img_pil)
#     fig,axs = plt.subplots(1,1, figsize=(18,15))
#     img_pil2.show(ctx=axs, title='superimposed')
#     semseg_img.show(ctx=axs, vmin=1, vmax=30);
#     buf = io.BytesIO()
#     fig.savefig(buf, format='png', dpi = 50)
#     buf.seek(0)
#     pil_img_out = deepcopy(Image.open(buf))
#     buf.close()
#     return
class SuperresHelper:
    @staticmethod
    def outImgFromPred(pred):
        img_out = pred[0]
        img_pil_out = PILImage.create(img_out)
        out_img_bytes=img_pil_out.to_bytes_format()
        return out_img_bytes

    @staticmethod
    def setup_dataloader(learn):
        dblock = DataBlock(blocks=(ImageBlock, ImageBlock),
                    get_items=get_image_files,
                    batch_tfms=[Normalize.from_stats(*imagenet_stats)])
        dbunch_mr = dblock.dataloaders(path, bs=1, val_bs=1, path=path, 
                        batch_tfms=[Normalize.from_stats(*imagenet_stats)])         
        dbunch_mr.c = 3        
        learn.dls = dbunch_mr
        return learn
