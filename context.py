from PIL import Image
import requests
from transformers import AutoProcessor, BlipModel
import os
import zarr
import torch
import numpy as np

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

dir = '/Datasets/MPHOI-72-raw/MPHOI_rgb'
save_dir = '/Datasets/MPHOI/mphoi_derived_features/global_context.zarr'
root = zarr.open(save_dir , mode='a')
chunk_size = [256, 256] 
dtype = np.float64 
compressor = {'id': 'gzip', 'level': 5}
compressor = zarr.get_codec(compressor) 
for subject in os.listdir(dir):
    dir_sub = os.path.join(dir, subject)
    for activity in os.listdir(dir_sub):
        dir_act = os.path.join(dir_sub, activity)
        for cnt in os.listdir(dir_act):
            name = '-'.join([subject, activity, cnt])
            if os.path.exists(os.path.join(save_dir, name)):
                continue
            print(name)
            data = []
            t = 0
            img_file = os.path.join(dir_act, cnt)
            for img_name in sorted(os.listdir(img_file)):
                img = Image.open(os.path.join(img_file, img_name))
                try:
                    img.verify()
                except:
                    pass
                else:
                    img = Image.open(os.path.join(img_file, img_name))
                    data.append(img)
                    t +=1
                if t == 100:
                    break
            inputs = processor(images=data, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.get_image_features(**inputs)
            ret = outputs.cpu().numpy()
            ds = root.create_dataset(
                     name,
                     shape=ret.shape,
                     chunks=chunk_size,
                     dtype=dtype,
                     compressor=compressor)
            ds[:] = ret
exit()
for i in range(1, 7):
    dir = '/Datasets/Bimanual_Actions/Bimanual_Actions_RGB-D_Dataset/bimacs_rgbd_data_subject_'+str(i)+'/bimacs_rgbd_data'
    save_dir = '/Datasets/BimanualActions/bimacs_derived_features/global_context.zarr'
    root = zarr.open(save_dir , mode='a')
    chunk_size = [256, 256] 
    dtype = np.float64 
    compressor = {'id': 'gzip', 'level': 5}
    compressor = zarr.get_codec(compressor) 
    for subject in os.listdir(dir):
        dir_sub = os.path.join(dir, subject)
        for activity in os.listdir(dir_sub):
            dir_act = os.path.join(dir_sub, activity)
            for cnt in os.listdir(dir_act):
                name = '-'.join([subject, activity, cnt])
                if os.path.exists(os.path.join(save_dir, name)):
                    continue
                print(name)
                dir_fin = os.path.join(dir_act, cnt, 'rgb')
                ret = []
                
                # img_file = dir_fin
                for chunk in sorted(os.listdir(dir_fin)):
                    data = []
                    img_file = os.path.join(dir_fin, chunk)
                    if not os.path.isdir(img_file):
                        continue
                    for img_name in sorted(os.listdir(img_file)):
                        img = Image.open(os.path.join(img_file, img_name))
                        try:
                            img.verify()
                        except:
                            pass
                        else:
                            img = Image.open(os.path.join(img_file, img_name))
                            data.append(img)
                        #     t +=1
                        # if t == 100:
                        #     break
                    inputs = processor(images=data, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = model.get_image_features(**inputs)
                    ret.append(outputs.cpu().numpy())
                ret = np.concatenate(ret, axis=0)
                ds = root.create_dataset(
                         name,
                         shape=ret.shape,
                         chunks=chunk_size,
                         dtype=dtype,
                         compressor=compressor)
                ds[:] = ret