import numpy as np
import glob
import pydicom
import os
import torch.utils.data as data
import torch

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



class DatasetFromFolder(data.Dataset):
    def __init__(self, input_image_array, target_image_array):
        super(DatasetFromFolder, self).__init__()
        
        self.input_image = input_image_array
        self.target_image = target_image_array


    def __getitem__(self, index):
        input = torch.from_numpy(self.input_image[index])
        target = torch.from_numpy(self.target_image[index])
        input = input.unsqueeze(0)
        target = target.unsqueeze(0)

        return input, target
    
    def __len__(self):
        return len(self.input_image)



def get_training_set():
    work_dir = "/data/CT/dataset/"
    train_dir = [work_dir + "L067/",
                work_dir + "L096/",
                work_dir + "L109/",
                work_dir + "L143/",
                work_dir + "L192/",
                work_dir + "L286/"]

    input_image_array = []
    target_image_array = []
    for index in range(0,len(train_dir)):
        input_file = train_dir[index] + "quarter_1mm/"
        target_file = train_dir[index] + "full_1mm/"
        input_image = get_data(input_file)
        target_image = get_data(target_file)

        c,h,w = input_image.shape
        print('loading data {}'.format(index))
        for i in range(0,c//3):
            for j in range(0,h//64):
                for k in range(0,w//64):
                    input_image_array.append(input_image[i*3:i*3+3,j*64:j*64+64,k*64:k*64+64])
                    target_image_array.append(target_image[i*3+1,j*64:j*64+64,k*64:k*64+64])

    return DatasetFromFolder(input_image_array,target_image_array)


def get_validation_set():
    work_dir = "/data/CT/dataset/"
    train_dir = [work_dir + "L310/",
                work_dir + "L333/"]

    input_image_array = []
    target_image_array = []
    for index in range(0,len(train_dir)):
        input_file = train_dir[index] + "quarter_1mm/"
        target_file = train_dir[index] + "full_1mm/"
        input_image = get_data(input_file)
        target_image = get_data(target_file)

        c,h,w = input_image.shape
        for i in range(0,c//3):
            for j in range(0,h//64):
                for k in range(0,w//64):
                    input_image_array.append(input_image[i*3:i*3+3,j*64:j*64+64,k*64:k*64+64])
                    target_image_array.append(target_image[i*3+1,j*64:j*64+64,k*64:k*64+64])

    return DatasetFromFolder(input_image_array,target_image_array)

def get_test_set():
    work_dir = "/data/CT/dataset/"
    train_dir = [work_dir + "L506/",
                work_dir + "L291/"]

    input_image_array = []
    target_image_array = []
    for index in range(0,len(train_dir)):
        input_file = train_dir[index] + "quarter_1mm/"
        target_file = train_dir[index] + "full_1mm/"
        input_image = get_data(input_file)
        target_image = get_data(target_file)

        c,h,w = input_image.shape
        for i in range(0,c//3):
            for j in range(0,h//64):
                for k in range(0,w//64):
                    input_image_array.append(input_image[i*3:i*3+3,j*64:j*64+64,k*64:k*64+64])
                    target_image_array.append(target_image[i*3+1,j*64:j*64+64,k*64:k*64+64])

    return DatasetFromFolder(input_image_array,target_image_array)


