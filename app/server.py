import aiohttp
from fastapi import FastAPI, File, UploadFile
from utils import *
from useless import * # Functions that are necessary to load the model but useless. Hopefully, fastai will fix this.
from superres import *
import asyncio
import uvicorn
from fastai2.vision.all import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse, FileResponse
from starlette.staticfiles import StaticFiles
import tempfile



path = Path(__file__).parent

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def async_setup_learner():
    print("Downloading model...")
    await download_file(export_file_url, path / export_file_name)
    print("Model downloaded")
    try:
        print("Loading model")
        learn = torch.load(path/export_file_name, map_location=torch.device('cpu'))
        print("Model loaded")
        learn.dls.device = 'cpu'
        learn=SuperresHelper.setup_dataloader(learn)
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


def img2img_do(img_bytes):
    # pred = learn.predict(img_bytes)
    # img_out = pred[0]
    # img_pil_out = PILImage.create(img_out)
    # out_img_bytes=img_pil_out.to_bytes_format()

    img_pil = Image.open(BytesIO(img_bytes))
    res_t = DivAndConqImg.predict(img_pil, learn)
    out_img_bytes = DivAndConqImg.outImgFromPred(res_t)

    with tempfile.NamedTemporaryFile(mode="w+b", suffix=".png", delete=False) as FOUT:
        FOUT.write(out_img_bytes)
        return FileResponse(FOUT.name, media_type="image/png")    

@app.post("/img2img/")
def img2img(file: UploadFile = File(...)):
    img_bytes = (file.file.read())
    return img2img_do(img_bytes)


@app.post("/urlimg2img/")
def urlimg2img(url: str):
    response = requests.get(url)
    return img2img_do(response.content)


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=80, log_level="info")
        
