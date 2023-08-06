## Virtual Try On WEB-APP using PF-AFN

Install Anaconda or Miniconda to create a new/separate environment  
```conda create -n tryon python=3.6```  
```source activate tryon ``` or ```conda activate tryon```  
```conda install pytorch=1.1.0 torchvision=0.3.0 cudatoolkit=9.0 -c pytorch```  
```conda install cupy``` or ```pip install cupy==6.0.0```  
```pip install opencv-python==4.5.1.48 fastapi "uvicorn[standard]" scipy python-multipart```

Download model checkpoints [[Checkpoints]](https://drive.google.com/file/d/1_a0AiN8Y_d_9TNDhHIcRlERz3zptyYWV/view?usp=sharing)  
Extract the zip file and copy the models into "checkpoints/PFAFN"  
It should look like "checkpoints/PFAFN/gen_model_final.pth" and "checkpoints/PFAFN/warp_model_final.pth"

Run the app
```
uvicorn app:app
```
Go To (http://localhost:8000)

