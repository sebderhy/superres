import aiohttp
from fastapi import FastAPI, File, UploadFile
import asyncio
import uvicorn
from fastai2.vision.all import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse, FileResponse
from starlette.staticfiles import StaticFiles
import tempfile

export_file_name = 'models/superres-1b.pth'

path = Path(__file__).parent

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def async_setup_learner():
    # await download_file(export_file_url, path / export_file_name)
    try:
        dblock = DataBlock(blocks=(ImageBlock, ImageBlock),
                   get_items=get_image_files,
                   get_y= lambda x:x,
                   batch_tfms=[Normalize.from_stats(*imagenet_stats)]
                   )
        dbunch_mr = dblock.dataloaders(path, bs=1, val_bs=1, path=path, batch_tfms=[Normalize.from_stats(*imagenet_stats)])         
        dbunch_mr.c = 3        

        learn = unet_learner(dbunch_mr, resnet34, loss_func=F.l1_loss, 
                     config=unet_config(blur=True, norm_type=NormType.Weight))
        learn.load(path/export_file_name)
        print("Model loaded")
        learn.dls.device = 'cpu'
        learn.dls = dbunch_mr
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(async_setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())



@app.post("/img2img/")
def img2img(file: UploadFile = File(...)):
    img_bytes = (file.file.read())
    pred = learn.predict(img_bytes)
    img_out = pred[0]
    img_pil_out = PILImage.create(img_out)
    # out_img_bytes = image_to_byte_array(img_pil_out)
    out_img_bytes=img_pil_out.to_bytes_format()
    with tempfile.NamedTemporaryFile(mode="w+b", suffix=".png", delete=False) as FOUT:
        FOUT.write(out_img_bytes)
        return FileResponse(FOUT.name, media_type="image/png")


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=80, log_level="info")
        
