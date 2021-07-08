import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
# import torchvision.models.resnet as resnet
from networks import resnet
from networks import resnext
import time
import os
from os import path
import random
from stl import mesh
import SimpleITK as sitk
import cv2
from datetime import datetime
import argparse
import tools
# import loss_functions
# from functions import mahalanobis
# from networks import generators
from networks import mynet
# from networks import p3d
# from networks import densenet
# from networks import autoencoder
# from networks import resnext
import sys
from skimage.measure._structural_similarity import compare_ssim as ssim
import gc
# import stn_test_center


################

desc = 'Training registration generator'
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('-i', '--init_mode',
                    type=str,
                    help="mode of training with different transformation matrics",
                    default='random_SRE2')

parser.add_argument('-t', '--training_mode',
                    type=str,
                    help="mode of training with different starting points",
                    default='scratch')

parser.add_argument('-m', '--model_filename',
                    type=str,
                    help="name of the pre-trained mode file",
                    default='None')

parser.add_argument('-l', '--learning_rate',
                    type=float,
                    help='Learning rate',
                    default=1e-4)

parser.add_argument('-d', '--device_no',
                    type=int,
                    choices=[0, 1, 2, 3, 4, 5, 6, 7],
                    help='GPU device number [0-7]',
                    default=0)

parser.add_argument('-e', '--epochs',
                    type=int,
                    help='number of training epochs',
                    default=200)

parser.add_argument('-n', '--network_type',
                    type=str,
                    help='choose different network architectures'
                         'the size of inputs/outputs are the same'
                         'could be original, resnext101',
                    default='mynet')

parser.add_argument('-info', '--information',
                    type=str,
                    help='infomation of this round of experiment',
                    default='Here is the information')

parser.add_argument('-ns', '--neighbour_slice',
                    type=int,
                    help='number of slice that acts as one sample',
                    default='5')

parser.add_argument('-it', '--input_type',
                    type=str,
                    help='input type of the network,'
                         'org_img, diff_img, optical flow',
                    default='org_img')

parser.add_argument('-ot', '--output_type',
                    type=str,
                    help='output type of the network,'
                         'average_dof, separate_dof, sum_dof',
                    default='average_dof')

pretrain_model_str = '0213-092230'

networks3D = ['resnext50', 'resnext101', 'densenet121', 'mynet', 'mynet2', 'p3d',
              'autoencoder', 'uda']

dof_stats = np.loadtxt('infos/label_stats.txt')
dof_means = np.mean(dof_stats, axis=0)
dof_std = np.std(dof_stats, axis=0)

net = 'Generator'
batch_size = 4
use_last_pretrained = False
current_epoch = 0

args = parser.parse_args()
device_no = args.device_no
epochs = args.epochs

training_progress = np.zeros((epochs, 4))
training_progress_new = []

hostname = os.uname().nodename
zion_common = '/zion/guoh9'
on_arc = False
if 'arc' == hostname:
    on_arc = True
    print('on_arc {}'.format(on_arc))
    # device = torch.device("cuda:{}".format(device_no))
    zion_common = '/raid/shared/guoh9'
    batch_size = 24
# device = torch.device("cuda:{}".format(device_no) if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:{}".format(device_no))
# print('start device {}'.format(device))

fan_mask = cv2.imread('data/avg_img.png', 0)

normalize_dof = True
# dof_stats = np.loadtxt('infos/dof_stats.txt')

def data_transform(input_img, crop_size=224, resize=224, normalize=False, masked_full=False):
    """
    Crop and resize image as you wish. This function is shared through multiple scripts
    :param input_img: please input a grey-scale numpy array image
    :param crop_size: center crop size, make sure do not contain regions beyond fan-shape
    :param resize: resized size
    :param normalize: whether normalize the image
    :return: transformed image
    """
    if masked_full:
        input_img[fan_mask == 0] = 0
        masked_full_img = input_img[112:412, 59:609]
        return masked_full_img

    h, w = input_img.shape
    if crop_size > 480:
        crop_size = 480
    x_start = int((h - crop_size) / 2)
    y_start = int((w - crop_size) / 2)

    patch_img = input_img[x_start:x_start+crop_size, y_start:y_start+crop_size]

    patch_img = cv2.resize(patch_img, (resize, resize))
    # cv2.imshow('patch', patch_img)
    # cv2.waitKey(0)
    if normalize:
        patch_img = patch_img.astype(np.float64)
        patch_img = (patch_img - np.min(patch_img)) / (np.max(patch_img) - np.mean(patch_img))

    return patch_img


def define_model(model_type, pretrained_path='', neighbour_slice=args.neighbour_slice,
                 input_type=args.input_type, output_type=args.output_type):
    if input_type == 'diff_img':
        input_channel = neighbour_slice - 1
    else:
        input_channel = neighbour_slice

    if model_type == 'prevost':
        model_ft = generators.PrevostNet()
    if model_type == 'autoencoder':
        model_ft = autoencoder.resnet50(sample_size=2, sample_duration=16, cardinality=32)
        model_ft.conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(3, 7, 7),
                                   stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
    elif model_type == 'resnext50':
        model_ft = resnext.resnet50(sample_size=2, sample_duration=16, cardinality=32)
        model_ft.conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(3, 7, 7),
                                   stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
    elif model_type == 'resnext101':
        model_ft = resnext.resnet101(sample_size=2, sample_duration=16, cardinality=32)
        model_ft.conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(3, 7, 7),
                                   stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        # model_ft.conv1 = nn.Conv3d(neighbour_slice, 64, kernel_size=7, stride=(1, 2, 2),
        #                            padding=(3, 3, 3), bias=False)
    elif model_type == 'resnet152':
        model_ft = resnet.resnet152(pretrained=True)
        model_ft.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2,
                                   padding=3, bias=False)
    elif model_type == 'resnet101':
        model_ft = resnet.resnet101(pretrained=True)
        model_ft.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2,
                                   padding=3, bias=False)
    elif model_type == 'resnet50':
        model_ft = resnet.resnet50(pretrained=True)
        model_ft.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2,
                                   padding=3, bias=False)
    elif model_type == 'resnet34':
        model_ft = resnet.resnet34(pretrained=False)
        model_ft.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2,
                                   padding=3, bias=False)
    elif model_type == 'resnet18':
        model_ft = resnet.resnet18(pretrained=True)
        model_ft.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2,
                                   padding=3, bias=False)
    elif model_type == 'mynet':
        model_ft = mynet.resnet50(sample_size=2, sample_duration=16, cardinality=32)
        model_ft.conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(3, 7, 7),
                                   stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
    elif model_type == 'mynet2':
        model_ft = generators.My3DNet()
    elif model_type == 'p3d':
        model_ft = p3d.P3D63()
        model_ft.conv1_custom = nn.Conv3d(1, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2),
                                          padding=(0, 3, 3), bias=False)
    elif model_type == 'densenet121':
        model_ft = densenet.densenet121()
    elif model_type == 'autoencoder':
        model_ft = autoencoder.resnet50()
    else:
        print('network type of <{}> is not supported, use original instead'.format(network_type))
        model_ft = generators.PrevostNet()

    num_ftrs = model_ft.fc.in_features

    if model_type == 'mynet':
        num_ftrs = 384
    elif model_type == 'prevost':
        num_ftrs = 576

    if output_type == 'average_dof' or output_type == 'sum_dof':
        # model_ft.fc = nn.Linear(128, 6)
        model_ft.fc = nn.Linear(num_ftrs, 6)
    else:
        # model_ft.fc = nn.Linear(128, (neighbour_slice - 1) * 6)
        model_ft.fc = nn.Linear(num_ftrs, (neighbour_slice - 1) * 6)



    # if args.training_mode == 'finetune':
    #     model_path = path.join(results_dir, args.model_filename)
    #     if path.isfile(model_path):
    #         print('Loading model from <{}>...'.format(model_path))
    #         model_ft.load_state_dict(torch.load(model_path))
    #         print('Done')
    #     else:
    #         print('<{}> not exists! Training from scratch...'.format(model_path))

    if pretrained_path:
        if path.isfile(pretrained_path):
            print('Loading model from <{}>...'.format(pretrained_path))
            model_ft.load_state_dict(torch.load(pretrained_path, map_location='cuda:0'))
            # model_ft.load_state_dict(torch.load(pretrained_path))
            print('Done')
        else:
            print('<{}> not exists! Training from scratch...'.format(pretrained_path))
    else:
        print('Train this model from scratch!')

    model_ft.cuda()
    model_ft = model_ft.to(device)
    print('define model device {}'.format(device))
    return model_ft


# input an image array
# normalize values to 0-255
def array_normalize(input_array):
    max_value = np.max(input_array)
    min_value = np.min(input_array)
    # print('max {}, min {}'.format(max_value, min_value))
    k = 255 / (max_value - min_value)
    min_array = np.ones_like(input_array) * min_value
    normalized = k * (input_array - min_array)
    return normalized


def filename_list(dir):
    images = []
    dir = os.path.expanduser(dir)
    print('dir {}'.format(dir))
    for filename in os.listdir(dir):
        # print(filename)
        file_path = path.join(dir, filename)
        images.append(file_path)
        # print(file_path)
    # print(images)
    return images


def normalize_volume(input_volume):
    # print('input_volume shape {}'.format(input_volume.shape))
    mean = np.mean(input_volume)
    std = np.std(input_volume)

    normalized_volume = (input_volume - mean) / std
    # print('normalized shape {}'.format(normalized_volume.shape))
    # time.sleep(30)
    return normalized_volume


def scale_volume(input_volume, upper_bound=255, lower_bound=0):
    max_value = np.max(input_volume)
    min_value = np.min(input_volume)

    k = (upper_bound - lower_bound) / (max_value - min_value)
    scaled_volume = k * (input_volume - min_value) + lower_bound
    # print('min of scaled {}'.format(np.min(scaled_volume)))
    # print('max of scaled {}'.format(np.max(scaled_volume)))
    return scaled_volume

def chooseFrame(frame_num, mid_ratio=0.5):
    mid_ratio = 0.5
    frame_start = int(frame_num * (1 -  mid_ratio) // 2)
    frame_end = int(frame_num * (1 + mid_ratio) // 2)
    # print('start {}, end {}'.format(frame_start, frame_end))
    random_id = random.randint(frame_start, frame_end) - 1
    # print(random_id)

    return random_id

class TestNetwork(Dataset):

    def __init__(self, case_id, frame_id, init_id=None):
        """
        """
        case_id = case_id

        case_recon_dir = path.join(recon_dir, case_id)
        case_frames_dir = path.join(us_dataset_dir, case_id, 'frames')
        case_mats = tools.readMatsFromSequence(case_id=case_id, on_arc=on_arc)
        clip_x, clip_y, clip_h, clip_w = 105, 54, 320, 565
        clip_info = [clip_x, clip_y, clip_h, clip_w]

        """ Load reconstructed _myrecon.mhd file into sitk img """
        case_recon_vol_fn = path.join(recon_dir, case_id, '{}_myrecon.mhd'.format(case_id))
        self.us_img_my = sitk.ReadImage(case_recon_vol_fn)
        self.raw_spacing = self.us_img_my.GetSpacing()
        sitksize = self.us_img_my.GetSize()
        vol_spacing = self.us_img_my.GetSpacing()

        """ Choose a random frame inside this sequence """
        frame_num = len(os.listdir(case_frames_dir))
        # frame_id = chooseFrame(frame_num=frame_num, mid_ratio=0.8)
        if not init_id:
            init_id = tools.chooseRandInit(frame_num=frame_num, frame_id=frame_id, rand_range=10)
        # frame_id, init_id = 50, 70
        frame_np = cv2.imread(path.join(case_frames_dir, '{:04}.jpg'.format(frame_id)), 0)
        frame_mat = case_mats[frame_id, :, :]
        frame_np = tools.processFrame(us_spacing=self.us_img_my.GetSpacing(), 
                                       frame_np=frame_np, frame_mat=frame_mat, clip_info=clip_info)
        # cv2.imwrite('tmp.jpg', frame_np)
        print('frame_np shape {}'.format(frame_np.shape))
        
        # sys.exit()

        """" Generate frame_mat and init_mat for initialization"""
        self.us_img_my.SetSpacing([1, 1, 1])
        self.frame_gt_mat = tools.matSitk2Stn(input_mat=case_mats[frame_id, :, :].copy(), 
                                  clip_size=(clip_x, clip_y), 
                                  raw_spacing=self.raw_spacing, frame_shape=frame_np.shape, 
                                  img_size=self.us_img_my.GetSize(), 
                                  img_spacing=self.us_img_my.GetSpacing(),
                                  img_origin=self.us_img_my.GetOrigin())
        self.init_mat = tools.matSitk2Stn(input_mat=case_mats[init_id, :, :].copy(), 
                                  clip_size=(clip_x, clip_y), 
                                  raw_spacing=self.raw_spacing, frame_shape=frame_np.shape, 
                                  img_size=self.us_img_my.GetSize(), 
                                  img_spacing=self.us_img_my.GetSpacing(),
                                  img_origin=self.us_img_my.GetOrigin())
        # print('frame_gt_mat\n{}'.format(self.frame_gt_mat))
        # print('init_mat\n{}'.format(self.init_mat))
        
        """ Initialize a subvolume using init_mat """
        # vol_size = (128, 128, 64)
        # frame_size = (128, 128)
        vol_size = (128, 128, 32)
        frame_size = (128, 128)
        init_vol = tools.sampleSubvol(sitk_img=self.us_img_my, init_mat=self.init_mat, 
                                      crop_size=vol_size)
        diff_mat = np.dot(np.linalg.inv(self.init_mat), self.frame_gt_mat)
        diff_dof = tools.mat2dof_np(input_mat=diff_mat)
        diff_dof_copy = diff_dof.copy()
        # print('diff_dof {}'.format(diff_dof))
        # print('frame_id {}'.format(frame_id))
        # print('frame_gt_mat\n{}'.format(self.frame_gt_mat))
        # print('origin {}'.format(self.us_img_my.GetOrigin()))
        # print('spacing {}'.format(self.us_img_my.GetSpacing()))
        slice_id = 0 + int(init_vol.shape[0] / 2)
        resampled_plane = init_vol[slice_id, :, :]
        cv2.imwrite('tmp_sample.jpg', resampled_plane)
        # print('saved')
        # sys.exit()

        init_vol = np.expand_dims(init_vol, 0)
        init_vol = np.expand_dims(init_vol, 0)
        us_tensor = torch.from_numpy(init_vol)
        us_tensor = us_tensor.float()
        # us_tensor = torch.from_numpy(np.expand_dims(init_vol, axis=(0, 1))).float()

        """ Trying to convert mat into 9DOF, without loss of generity """
        mat_np = tools.dof2mat_np(input_dof=diff_dof, scale=False)
        recon_diff = np.dot(np.linalg.inv(diff_mat), mat_np)
        # print('recon_diff\n{}'.format(recon_diff))

        # mat_ext = np.expand_dims(self.init_mat, axis=0)
        # mat_ext = np.expand_dims(mat_np, axis=0)
        mat_ext = np.expand_dims(diff_mat, axis=0)
        mat_tensor = torch.tensor(mat_ext)
        frame_crop = tools.frameCrop(input_np=frame_np, crop_size=frame_size)

        """ Apply stn to sample US_Vol """
        # grid = tools.myAffineGrid2(input_tensor=us_tensor, input_mat=mat_tensor, 
        #                            input_spacing=self.us_img_my.GetSpacing())
        # us_tensor_resampled = F.grid_sample(us_tensor, grid, align_corners=True)
        # resampled_array = us_tensor_resampled.squeeze().numpy().copy()
        # slice_id = 0 + int(resampled_array.shape[0] / 2)
        # resampled_plane = resampled_array[slice_id, :, :]
        # resampled_crop = tools.frameCrop(input_np=resampled_plane, crop_size=frame_size)
        # diff = resampled_crop - frame_crop
        # print('max {}, min {}, mean {}'.format(np.max(diff), np.min(diff), np.mean(diff)))
        # concat = np.concatenate((frame_crop, resampled_crop, diff), axis=0)
        # cv2.imwrite('tmp.jpg', concat)
        # sys.exit()

        frame_tensor = np.expand_dims(frame_crop, axis=0)
        frame_tensor = torch.from_numpy(frame_tensor)
        self.frame_tensor = frame_tensor.unsqueeze(0)
        self.vol_tensor = us_tensor.squeeze(0)
        # mat_tensor = torch.from_numpy(diff_mat)
        self.mat_tensor = torch.from_numpy(mat_np)
        # print('diff_dof loader\n{}'.format(diff_dof_copy))
        self.dof_tensor = torch.from_numpy(diff_dof_copy[:6])

        self.vol_tensor = self.vol_tensor.to(device)
        self.frame_tensor = self.frame_tensor.to(device)
        # mat_tensor = mat_tensor.to(device)
        # dof_tensor = dof_tensor.to(device)

        self.vol_tensor = self.vol_tensor.unsqueeze(0).float()
        self.frame_tensor = self.frame_tensor.unsqueeze(0).float()
        self.mat_tensor = self.mat_tensor.unsqueeze(0).float()

        start = time.time()
        vol_resampled, mat_16 = model_ft(vol=self.vol_tensor, frame=self.frame_tensor, device=device)
        self.time_cost = time.time() - start

        np.set_printoptions(precision=2)
        # self.dof_predi = self.dof_tensor.cpu().detach().squeeze().numpy()
        # if normalization:
        #     self.dof_label = (mat_16.cpu().detach().squeeze().numpy()*dof_std+dof_means)
        # else:
        #     self.dof_label = mat_16.cpu().detach().squeeze().numpy()
        self.dof_label = self.dof_tensor.cpu().detach().squeeze().numpy()
        if normalization:
            self.dof_predi = (mat_16.cpu().detach().squeeze().numpy()*dof_std+dof_means)
        else:
            self.dof_predi = mat_16.cpu().detach().squeeze().numpy()
        # self.dof_predi = tools.generateRandomGuess(dof_means, dof_std)
        print('*** Test with {} frame_id {}, init_id {} ***'.format(case_id, frame_id, init_id))
        print('Label: {}'.format(self.dof_label))
        print('Predi: {}'.format(self.dof_predi))

        mat_label = tools.dof2mat_np(input_dof=self.dof_label)
        mat_predi = tools.dof2mat_np(input_dof=self.dof_predi)
        # mat_predi = tools.dof2mat_np(input_dof=np.asarray([0, 0, 0, 0, 0, 0]))
        mat_error = np.dot(np.linalg.inv(mat_label), mat_predi)
        # del self.dof_label, self.dof_predi
        print('mat_error\n{}'.format(mat_error))
        self.error_mm = tools.computeError(mat_error=mat_error, spacing=self.raw_spacing[0], 
                                           img_size=frame_crop.shape)
        # self.error_mm = tools.computeError(mat_error=diff_mat, spacing=self.raw_spacing[0],
        #                                    img_size=frame_crop.shape)
        print('error {:.4f} mm'.format(self.error_mm))

        grid = tools.myAffineGrid2(input_tensor=self.vol_tensor, input_mat=self.mat_tensor, 
                                   input_spacing=(1, 1, 1))
        grid = grid.to(device)
        us_tensor_resampled = F.grid_sample(self.vol_tensor, grid, align_corners=True)
        gt_np = us_tensor_resampled.squeeze().cpu().numpy().copy()
        # print('gt_np shape {}'.format(gt_np.shape))

        slice_id = int(self.vol_tensor.shape[2] / 2)
        # print('slice_id {}'.format(slice_id))

        out_np = vol_resampled.cpu().squeeze().detach().numpy()[slice_id, :, :]
        frame_np = self.frame_tensor.cpu().squeeze().detach().numpy()
        gt_np = gt_np[slice_id, :, :]
        self.img_sim = ssim(gt_np, out_np)
        print('img_sim {:.4f}'.format(self.img_sim))
        self.img_cor = tools.correlation_coefficient(gt_np, out_np)
        print('img_cor {:.4f}\n'.format(self.img_cor))
        # sys.exit()
        # print('{} out {}, frame {}'.format(frame_id, out_np.shape, frame_np.shape))
        cat_np = np.concatenate((frame_np, gt_np, out_np), axis=0)
        img_res_fn = path.join(case_res_reg_dir, '{}_{}.jpg'.format(case_id, frame_id))
        # img_res_fn = 'tmp_res.jpg'
        cv2.imwrite(img_res_fn, cat_np)
        cv2.imwrite('tmp_stn.jpg', cat_np)


def defineModel(model_type, model_str=None, device=torch.device("cuda:0")):
    pretrain_model_str = model_str
    # pretrain_model_str = 'mynet3_101'  # mynet3 resnext 101 on arc
    # pretrain_model_str = 'mynet3_150'  # mynet3 resnext 150 on arc
    # pretrain_model_str = 'mynet3_150_l1'  # mynet3 resnext 150 on arc
    # pretrain_model_str = 'mynet4_150'  # deep branch

    model_folder = '/zion/guoh9/US_recon/results'
    # model_ft = mynet.mynet3()
    # model_ft = mynet.mynet4()

    if model_type == 'mynet3_50':
        model_ft = mynet.mynet3(layers=[3, 4, 6, 3])
    elif model_type == 'mynet3_101':
        model_ft = mynet.mynet3(layers=[3, 4, 23, 3])
    elif model_type == 'mynet3_150':
        model_ft = mynet.mynet3(layers=[3, 8, 36, 3])
    elif model_type == 'mynet4_150':
        model_ft = mynet.mynet4()
    else:
        print('Network type {} not supported!'.format(model_type))
        sys.exit()
    
    if pretrain_model_str:
        model_path = path.join(model_folder, '3d_best_Generator_{}.pth'.format(pretrain_model_str))  # 10
        model_ft.load_state_dict(torch.load(model_path, map_location='cuda:0'))

    model_ft.cuda()
    model_ft.eval()
    model_ft = model_ft.to(device)

    return model_ft




if __name__ == '__main__':
    # data_dir = path.join('/home/guoh9/tmp/US_vid_frames')
    # results_dir = path.join('/home/guoh9/tmp/US_vid_frames')

    # data_dir = path.join(zion_common, 'US_recon/US_vid_frames')
    # pos_dir = path.join(zion_common, 'US_recon/US_vid_pos')
    uronav_dir = path.join(zion_common, 'uronav_data')
    us_dataset_dir = path.join(zion_common, 'US_recon/US_dataset')

    recon_dir = path.join(zion_common, 'US_recon/recon')
    seq_dir = path.join(zion_common, 'US_recon/new_data')

    # train_ids = np.loadtxt('infos/train_ids.txt')
    # val_ids = np.loadtxt('infos/val_ids.txt')
    # clean_ids = {'train': train_ids, 'val': val_ids}
    # print('train_ids\n{}'.format(train_ids))
    # sys.exit()

    if 'arc' == hostname:
        results_dir = '/home/guoh9/US_recon/results'
    else:
        results_dir = path.join(zion_common, 'US_recon/results')

    init_mode = args.init_mode
    network_type = args.network_type
    print('Transform initialization mode: {}'.format(init_mode))
    print('Training mode: {}'.format(args.training_mode))

    dataset_dir = 'infos/sets/all_cases'
    # dataset_dir = 'infos/sets/spacing0.5'
    # dataset_dir = 'infos/sets/spacing0.5_tiny'
    print(os.listdir(dataset_dir))
    
    model_type, model_str = 'mynet3_50', '0226-035203' # mynet3 50 no norm
    # model_type, model_str = 'mynet3_50', '0225-213419' # mynet3 50 norm
    # model_type, model_str = 'mynet3_50', '0226-131924' # mynet3 101 norm

    model_type, model_str = 'mynet3_150', 'test' # mynet3 150 no norm
    # model_type, model_str = 'mynet3_150', '0302-142027' # mynet3 150 norm
    # model_type, model_str = 'mynet3_150', '150norm' # mynet3 150 norm
    # model_type, model_str = 'mynet3_150', '150nonorm' # mynet3 150 nonorm
    
    # model_type, model_str = 'mynet3_101', 'mynet3_101' # mynet3 101 no norm
    # model_type, model_str = 'mynet3_101', '0302-025557' # mynet3 101 no norm
    # model_type, model_str = 'mynet3_101', '0302-142419' # mynet3 101 no norm

    model_ft = defineModel(model_type, model_str)
    model_ft.to(device)

    # model_str = 'guess'
    normalization = False
    # normalization = True
    
    test_ids = np.loadtxt(path.join(dataset_dir, 'test.txt'))

    case_id = 'Case0009'

    case_frames_dir = path.join(us_dataset_dir, case_id, 'frames')
    frame_num = len(os.listdir(case_frames_dir))
    print('{} has {} frames'.format(case_id, frame_num))

    case_res_dir = 'results/{}'.format(case_id)
    if not path.isdir(case_res_dir):
        os.mkdir(case_res_dir)
    case_res_reg_dir = path.join(case_res_dir, 'regres')
    if not path.isdir(case_res_reg_dir):
        os.mkdir(case_res_reg_dir)

    mid_id = frame_num // 2
    mid_id = 58
    lower_range, upper_range = mid_id - 10, mid_id + 11
    print('mid_id {}, range ({}, {})'.format(mid_id, lower_range, upper_range))

    preds = []
    labels = []

    errors_mm = []
    img_sims = []
    img_cors = []
    times = []
    

    for i in range(lower_range, upper_range):
        frame_id = i

        # model_ft = defineModel(model_type, model_str)
        case = TestNetwork(case_id=case_id, frame_id=frame_id, init_id=mid_id)

        # case = stn_test_center.Slice2Volume(case_id='Case0009', frame_id=frame_id, 
        #                                     initial_tfm=None)

        preds.append(case.dof_predi)
        labels.append(case.dof_label)

        preds2 = np.asarray(preds)
        labels2 = np.asarray(labels)
        labels_preds2 = np.concatenate((labels2, preds2), axis=1)
        np.savetxt(path.join(case_res_dir, 'labels_preds_{}.txt'.format(model_str)), labels_preds2)
        np.savetxt(path.join(case_res_dir, 'labels_preds_{}_{}.txt'.format(model_str, mid_id)), labels_preds2)

        errors_mm.append(case.error_mm)
        np.savetxt(path.join(case_res_dir, 'errors_mm_{}.txt'.format(model_str)), np.asarray(errors_mm))

        img_cors.append(case.img_cor)
        np.savetxt(path.join(case_res_dir, 'img_cors_{}.txt'.format(model_str)), np.asarray(img_cors))

        times.append(case.time_cost)
        np.savetxt(path.join(case_res_dir, 'times_{}.txt'.format(model_str)), np.asarray(times))

        
        # del case, model_ft
        # torch.cuda.empty_cache()
        # gc.collect()
    print('Avg_time {:.4f} s'.format(np.mean(np.asarray(times))))
    print('Avg_err {:.4f} mm'.format(np.mean(np.asarray(errors_mm))))
    print('Avg_cor {:.4f}'.format(np.mean(np.asarray(img_cors))))
        
    # preds = np.asarray(preds)
    # labels = np.asarray(labels)
    # labels_preds = np.concatenate((labels, preds), axis=1)
    # np.savetxt('labels_preds.txt', labels_preds)
        




