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

class DivAndConqImg:
    @staticmethod
    def predict(img_pil, learn, buffer_lines = 20):
        t = TensorImage(img_pil)
        t2 = evenify(t)
        ul,bl,ur,br=DivAndConqImg.split_tensimg_in_4(t2, buffer_lines)
        crops = [ul,bl,ur,br]
        decs = DivAndConqImg.predict_on_img_list(crops,learn)
        res_t = DivAndConqImg.merge_to_output(decs,t2.shape[0],t2.shape[1], buffer_lines)
        return res_t
    
    @staticmethod
    def split_tensimg_in_4(t, buf, channel_first=False):
        rows,cols,_ = t.shape
        if channel_first:
            ul=t[:rows//2+buf,:cols//2+buf,:].permute(2,0,1)
            bl=t[rows//2-buf:rows,:cols//2+buf,:].permute(2,0,1)
            ur=t[:rows//2+buf,cols//2-buf:cols,:].permute(2,0,1)
            br=t[rows//2-buf:rows,cols//2-buf:cols,:].permute(2,0,1)
        else:
            ul=t[:rows//2+buf,:cols//2+buf,:]
            bl=t[rows//2-buf:rows,:cols//2+buf,:]
            ur=t[:rows//2+buf,cols//2-buf:cols,:]
            br=t[rows//2-buf:rows,cols//2-buf:cols,:]
        return ul,bl,ur,br

    @staticmethod
    def stack_4_images_into_batch(ul,bl,ur,br):
        batch = torch.stack([ul,bl,ur,br])
        print(batch.shape)
        return batch

    @staticmethod
    def predict_on_img_list(items,learn):
        seb_dl = learn.dls.test_dl(items, rm_type_tfms=None)
        inp,_,_,dec_preds = learn.get_preds(dl=seb_dl, with_input=True, with_decoded=True)
        decs = learn.dls.decode_batch((*tuplify(inp),*tuplify(dec_preds)))[:]
        return decs

    @staticmethod
    def merge_to_output(decs, rows, cols, buf):
        res_t = torch.empty(3,rows,cols)
        res_t[:,:rows//2,:cols//2]=decs[0][1][:,:rows//2,:cols//2]
        res_t[:,rows//2:rows,:cols//2]=decs[1][1][:,buf:,:cols//2]
        res_t[:,:rows//2,cols//2:cols]=decs[2][1][:,:rows//2,buf:]
        res_t[:,rows//2:rows,cols//2:cols]=decs[3][1][:,buf:,buf:]
        return res_t

    @staticmethod
    def outImgFromPred(res_t):
        pil_img_out = PILImage.create(TensorImage(res_t.byte()))
        out_img_bytes = pil_img_out.to_bytes_format()
        return out_img_bytes