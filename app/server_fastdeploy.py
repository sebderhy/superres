from utils import *
from useless import * # Functions that are necessary to load the model but useless. Hopefully, fastai will fix this.
from superres import *
from semseg import *
from fastapi import FastAPI, File, UploadFile
import asyncio
import uvicorn
from fastai2.vision.all import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
import tempfile


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

# loop = asyncio.get_event_loop()
# tasks = [asyncio.ensure_future(async_setup_learner())]
# learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
# loop.close()

def setup_learner(model_name: str):
    try:
        learn = torch.load(path/f'models/{model_name}.pkl', map_location=torch.device('cpu'))
        # learn = load_learner(path/f'models/{model_name}.pkl')
        learn.dls.device = 'cpu'
        print("model loaded")
        if(model_name.startswith('superres')):
            learn=SuperresHelper.setup_dataloader(learn)
            print("Superres data loaders configured")
        if(model_name.startswith('semseg')):
            learn=SuperresHelper.setup_dataloader(learn)
            print("Semseg data loaders configured")
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            print(e)
            raise


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())
    
def img_predict_do(model_name: str, img_bytes: bytes):
    learn = setup_learner(model_name)
    pred = safe_predict(learn, img_bytes)
    json_resp = JSONResponse({
        'result': str(pred[0])
    })
    del learn    
    return json_resp


@app.post("/{model_name}/img2class/")
def img2class(model_name: str, file: UploadFile = File(...)):
    img_bytes = (file.file.read())
    return img_predict_do(model_name, img_bytes)


@app.post("/{model_name}/urlimg2class/")
def urlimg2class(model_name: str, url: str):
    response = requests.get(url)
    return img_predict_do(model_name, response.content)


def img2img_do(model_name: str, img_bytes: bytes):
    if(len(img_bytes)>2e6):
        return JSONResponse({
                'error': "Sorry! Images larger than 2MB cannot be handled at the moment."
        }) 
    learn = setup_learner(model_name)
    img_pil = PILImage.create(img_bytes)
    try:
        print("predicting")
        pred = learn.predict(img_pil)    
        print("prediction done")
        # if(model_name=='is-img-rotated'):
        #     img_pil = Image.open(BytesIO(img_bytes))
        #     img_pil_out = derotate_img(pred, img_pil)
        #     out_img_bytes = image_to_byte_array(img_pil_out)
        if(model_name.startswith('superres')):
            out_img_bytes = SuperresHelper.outImgFromPred(pred)
            print("High Res Img computed")
        if(model_name.startswith('semseg')):
            img_pil = Image.open(BytesIO(img_bytes))
            out_img_bytes= SemsegHelper.outImgFromPred(pred,img_pil)        
        del learn
    except RuntimeError as e:
        print(e)
        del learn
        return JSONResponse({
            'error': e
        }) 
    return bytes2out(out_img_bytes)


@app.post("/{model_name}/img2img/")
def img2img(model_name: str, file: UploadFile = File(...)):
    img_bytes = (file.file.read())
    return img2img_do(model_name,img_bytes)


@app.post("/{model_name}/urlimg2img/")
def urlimg2img(model_name: str, url: str):
    response = requests.get(url)
    return img2img_do(model_name, response.content)


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=80, log_level="info")
        
