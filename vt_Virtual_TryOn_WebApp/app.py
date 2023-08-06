from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import os
import shutil
import subprocess

from prepare_data import generate_masks, generate_pairs


def create_dirs():
    if os.path.exists("results"):
        shutil.rmtree("results")
    if os.path.exists("dataset"):
        shutil.rmtree("dataset")

    os.makedirs("results")
    os.makedirs("dataset")
    os.makedirs("dataset/test_img")
    os.makedirs("dataset/test_clothes")
    os.makedirs("dataset/test_edge")


app = FastAPI()
create_dirs()

# Mount the 'static' directory to serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_index():
    # Serve the index.html file
    with open("static/index.html", "r") as f:
        return f.read()


@app.post("/upload/person/")
async def upload_person_image(image: UploadFile = File(...)):
    return await save_image(image, "test_img")


@app.post("/upload/clothing/")
async def upload_clothing_image(image: UploadFile = File(...)):
    return await save_image(image, "test_clothes")


async def save_image(image: UploadFile, folder: str):
    save_path = os.path.join("dataset", folder)
    file_path = os.path.join(save_path, image.filename)

    img = Image.open(image.file)
    img = img.resize((192, 256))

    # Save the resized image
    with open(file_path, "wb") as f:
        img.save(f, format="JPEG")


@app.post("/virtual_try_on_all/")
async def run_virtual_try_on_all():
    generate_masks()
    generate_pairs()
    subprocess.run("python test.py --name demo --resize_or_crop None --batchSize 1 --gpu_ids 0", shell=True)
    return {"result_images": os.listdir("results/demo/PFAFN")}


@app.get("/get_result_image/{image_name}")
async def get_result_image(image_name: str):
    image_path = os.path.join("results/demo/PFAFN", image_name)
    return FileResponse(image_path)


@app.post("/reset/")
async def reset_images():
    create_dirs()
    
