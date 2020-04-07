from fastai2.vision.all import *
import aiohttp
from starlette.responses import HTMLResponse, JSONResponse, FileResponse

export_file_url = 'https://fastdeploy2.s3.amazonaws.com/fastai-models/superres-2b.pkl' 
export_file_name = 'models/superres-2b.pkl'

path = Path(__file__).parent

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


def image_to_byte_array(image:Image):
  imgByteArr = io.BytesIO()
  image.save(imgByteArr, format=image.format)
  imgByteArr = imgByteArr.getvalue()
  return imgByteArr


def derotate_img(pred, img_pil: Image):
    img_pil_out = img_pil
    rotation_state = str(pred[0])
    if(rotation_state=='rotated180'): img_pil_out = img_pil.rotate(180)
    if(rotation_state=='rotated90'): img_pil_out = img_pil.rotate(-90)
    if(rotation_state=='rotated270'): img_pil_out = img_pil.rotate(90)
    img_pil_out.format = img_pil.format
    return img_pil_out

def bytes2out(out_img_bytes):
    with tempfile.NamedTemporaryFile(mode="w+b", suffix=".png", delete=False) as FOUT:
        FOUT.write(out_img_bytes)
        return FileResponse(FOUT.name, media_type="image/png")


def safe_predict(learn,img_bytes):
    try:
        return learn.predict(img_bytes)
    except RuntimeError as e:
        print(e)
        raise e   