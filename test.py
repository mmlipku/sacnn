import glob
import argparse, os
import pydicom
import torch
from torch.autograd import Variable
from PIL import Image
import numpy as np
import time, math
import cv2
import skimage

parser = argparse.ArgumentParser(description="Pytorch demo")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="checkpoint/model_epoch_100.pth", type=str, help="model path")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")

def PSNR(pred, gt):
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


def read_dicom_scan(scan_path):
    dicom_files = glob.glob(os.path.join(scan_path,'*.IMA'))
    slices = [pydicom.read_file(each_dicom_path) for each_dicom_path in dicom_files]

    slices.sort(key = lambda x: float(x.InstanceNumber))
    if len(slices) == 0:
        print('Scan reading error, please check the scan path')
    
    return(slices)


def get_pixels_HU(slices):

    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)
                      
    return np.array(image, dtype=np.int16)


def get_data(scan_path):
    
    test_slices = read_dicom_scan(scan_path)
    image = get_pixels_HU(test_slices)

    img_max = np.max(image)
    img_min = np.min(image)

    crop_max = 240
    crop_min = -160

    image = np.clip(image,crop_min,crop_max)
    nor_image = image.astype('float32')
    nor_image = (nor_image - crop_min)/(crop_max - crop_min)
    return nor_image


opt = parser.parse_args()
cuda = opt.cuda

if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]

input_file = "/data/CT/L291/quarter_1mm/"
target_file = "/data/CT/L291/full_1mm/"
input_image = get_data(input_file)
target_image = get_data(target_file)

input_image = input_image[368-1:368+2,120:248,110:238]
target_image = target_image[368,120:248,110:238]


avg_elapsed_time = 0.0


im_input = Variable(torch.from_numpy(input_image).float()).view(1, -1, input_image.shape[0], input_image.shape[1],input_image.shape[2])
print(im_input.shape)

if cuda:
    model = model.cuda()
    im_input = im_input.cuda()

start_time = time.time()
#out,p1,p2,p3 = model(im_input)
out = model(im_input)
elapsed_time = time.time() - start_time
avg_elapsed_time += elapsed_time

out = out.cpu()
im_out = out.data[0].numpy().astype(np.float32)
im_output = im_out[0,:,:]

#feature = p1.data[0].numpy().astype(np.float32)
#print(feature.shape)

input_image = input_image[1,:,:] * 255.
output_image = im_output * 255.
output_image = np.clip(output_image,0,255.)
target_image = target_image * 255.

psnr_predicted = PSNR(output_image, target_image)
score = cv2.compareHist(np.array(output_image, dtype=np.float32), np.array(target_image, dtype=np.float32), cv2.HISTCMP_CORREL)
psnr = skimage.measure.compare_ssim(output_image/255, target_image/255)

input = Image.fromarray(np.int8(input_image), "L")
output = Image.fromarray(np.int8(output_image), "L")
original = Image.fromarray(np.int8(target_image), "L")
#fea = Image.fromarray(np.int8(feature), "L")

print("PSNR_predicted=", psnr_predicted)
print("It takes {}s for processing".format(elapsed_time))
print("Score is :", score)  
print("SSIM is :", psnr)  

input.save('result/in.bmp')
output.save('result/out.png')
original.save('result/org.bmp')
#fea.save('result/feature.bmp')

