import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
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
from networks import fvrnet
import sys
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from scipy import stats
import test_network2
################

desc = 'Training registration generator'
parser = argparse.ArgumentParser(description=desc)

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
                    default=1)

parser.add_argument('-e', '--epochs',
                    type=int,
                    help='number of training epochs',
                    default=150)

parser.add_argument('-n', '--network_type',
                    type=str,
                    help='choose different network architectures'
                         'the size of inputs/outputs are the same'
                         'could be original, resnext101',
                    default='mynet')


pretrain_model_str = '0213-092230'

networks3D = ['resnext50', 'resnext101', 'densenet121', 'mynet', 'mynet2', 'p3d',
              'autoencoder', 'uda']

net = 'Generator'
# batch_size = 2
batch_size = 16
use_last_pretrained = False
current_epoch = 0

dof_stats = np.loadtxt('infos/label_stats.txt')
dof_means = np.mean(dof_stats, axis=0)
dof_std = np.std(dof_stats, axis=0)
# print('dof_means\n{}'.format(dof_means))
# print('dof_std\n{}'.format(dof_std))

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
    batch_size = 40
# device = torch.device("cuda:{}".format(device_no) if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:{}".format(device_no))
# print('start device {}'.format(device))

fan_mask = cv2.imread('data/avg_img.png', 0)

# normalize_dof = True
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


def defineModel(model_type):
    pretrain_model_str = model_type

    if model_type == 'mynet3_50':
        model_ft = mynet.mynet3(layers=[3, 4, 6, 3])
    elif model_type == 'mynet3_101' or model_type == 'test':
        model_ft = mynet.mynet3(layers=[3, 4, 23, 3])
    elif model_type == 'mynet3_150' or 'mynet3_150_l1':
        model_ft = mynet.mynet3(layers=[3, 8, 36, 3])
    elif model_type == 'mynet4_150':
        model_ft = mynet.mynet4()
    else:
        print('Network type {} not supported!'.format(model_type))
        sys.exit()
    
    # model_path = path.join(model_folder, '3d_best_Generator_{}.pth'.format(pretrain_model_str))  # 10
    # model_ft.load_state_dict(torch.load(model_path, map_location='cuda:0'))
    model_ft.cuda()
    model_ft.eval()
    model_ft = model_ft.to(device)

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

class FreehandUS4D(Dataset):

    def __init__(self, root_dir, transform=None):
        """
        """
        samples = np.loadtxt(root_dir)
        self.samples = samples
        self.transform = transform
        self.phase = root_dir.split('/')[-1].split('.')[0]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        case_id = 'Case{:04}'.format(int(self.samples[idx]))
        # print('{} in {}'.format(case_id, self.phase))
        # case_id = 'Case0193'
        # case_id = 'Case0210'
        # case_id = 'Case0625'
        # case_id = 'Case0005'
        # case_id = 'Case0004'

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
        frame_id = chooseFrame(frame_num=frame_num, mid_ratio=0.8)

        if self.phase == 'train':
            init_id = tools.chooseRandInit(frame_num=frame_num, frame_id=frame_id, rand_range=10)
        else:
            # init_id = random.choice([frame_id-5, frame_id+5])
            # while init_id <= 0 or init_id >= frame_num:
            #     init_id = random.choice([frame_id-5, frame_id+5])
            init_id = tools.chooseRandInit(frame_num=frame_num, frame_id=frame_id, rand_range=10)

        # frame_id, init_id = 50, 70
        # print('frame_id {}, init_id {}'.format(frame_id, init_id))
        frame_np = cv2.imread(path.join(case_frames_dir, '{:04}.jpg'.format(frame_id)), 0)
        frame_mat = case_mats[frame_id, :, :]
        frame_np = tools.processFrame(us_spacing=self.us_img_my.GetSpacing(), 
                                       frame_np=frame_np, frame_mat=frame_mat, clip_info=clip_info)
        # cv2.imwrite('tmp.jpg', frame_np)
        # print('frame_np shape {}'.format(frame_np.shape))
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
        frame_tensor = frame_tensor.unsqueeze(0)
        vol_tensor = us_tensor.squeeze(0)
        # mat_tensor = torch.from_numpy(diff_mat)
        mat_tensor = torch.from_numpy(mat_np)
        # print('diff_dof loader\n{}'.format(diff_dof_copy))
        # print('before norm\n{}'.format(diff_dof_copy[:6]))
        # print('after norm\n{}'.format((diff_dof_copy[:6]-dof_means)/dof_std))
        if not normalize_dof:
            dof_tensor = torch.from_numpy(diff_dof_copy[:6])
            # print('no norm')
        else:
            dof_tensor = torch.from_numpy((diff_dof_copy[:6]-dof_means)/dof_std)
        # print('dof_tensor loader\n{}'.format(dof_tensor))
        # print('***** {} *****'.format(case_id))
        # print('vol_tensor shape {}'.format(vol_tensor.shape))
        # print('frame_tensor shape {}'.format(frame_tensor.shape))
        # print('mat_tensor shape {}'.format(mat_tensor.shape))
        # print('dof_tensor shape {}'.format(dof_tensor.shape))
        # print(diff_dof_copy[:6])
        # print('\n')
        # sys.exit()
        frameid_tensor = torch.from_numpy(np.asarray([frame_id, init_id]))


        return case_id, vol_tensor, frame_tensor, mat_tensor, dof_tensor, frameid_tensor



def get_dist_loss(labels, outputs, start_params, calib_mat):
    # print('labels shape {}'.format(labels.shape))
    # print('outputs shape {}'.format(outputs.shape))
    # print('start_params shape {}'.format(start_params.shape))
    # print('calib_mat shape {}'.format(calib_mat.shape))

    # print('labels_before\n{}'.format(labels.shape))
    labels = labels.data.cpu().numpy()
    outputs = outputs.data.cpu().numpy()
    if normalize_dof:
        labels = labels / dof_stats[:, 1] + dof_stats[:, 0]
        outputs = outputs / dof_stats[:, 1] + dof_stats[:, 0]


    start_params = start_params.data.cpu().numpy()
    calib_mat = calib_mat.data.cpu().numpy()

    if args.output_type == 'sum_dof':
        batch_errors = []
        for sample_id in range(labels.shape[0]):
            gen_param = tools.get_next_pos(trans_params1=start_params[sample_id, :],
                                           dof=outputs[sample_id, :],
                                           cam_cali_mat=calib_mat[sample_id, :, :])
            gt_param = tools.get_next_pos(trans_params1=start_params[sample_id, :],
                                          dof=labels[sample_id, :],
                                          cam_cali_mat=calib_mat[sample_id, :, :])
            gen_param = np.expand_dims(gen_param, axis=0)
            gt_param = np.expand_dims(gt_param, axis=0)

            result_pts = tools.params2corner_pts(params=gen_param, cam_cali_mat=calib_mat[sample_id, :, :])
            gt_pts = tools.params2corner_pts(params=gt_param, cam_cali_mat=calib_mat[sample_id, :, :])

            sample_error = tools.evaluate_dist(pts1=gt_pts, pts2=result_pts)
            batch_errors.append(sample_error)
        batch_errors = np.asarray(batch_errors)

        avg_batch_error = np.asarray(np.mean(batch_errors))
        error_tensor = torch.tensor(avg_batch_error, requires_grad=True)
        error_tensor = error_tensor.type(torch.FloatTensor)
        error_tensor = error_tensor.to(device)
        error_tensor = error_tensor * 0.99
        # print('disloss device {}'.format(device))
        # print(error_tensor)
        # time.sleep(30)
        return error_tensor




    if args.output_type == 'average_dof':
        labels = np.expand_dims(labels, axis=1)
        labels = np.repeat(labels, args.neighbour_slice - 1, axis=1)
        outputs = np.expand_dims(outputs, axis=1)
        outputs = np.repeat(outputs, args.neighbour_slice - 1, axis=1)
    else:
        labels = np.reshape(labels, (labels.shape[0], labels.shape[1] // 6, 6))
        outputs = np.reshape(outputs, (outputs.shape[0], outputs.shape[1] // 6, 6))
    # print('labels_after\n{}'.format(labels.shape))
    # print('outputs\n{}'.format(outputs.shape))
    # time.sleep(30)

    batch_errors = []
    final_drifts = []
    for sample_id in range(labels.shape[0]):
        gen_param_results = []
        gt_param_results = []
        for neighbour in range(labels.shape[1]):
            if neighbour == 0:
                base_param_gen = start_params[sample_id, :]
                base_param_gt = start_params[sample_id, :]
            else:
                base_param_gen = gen_param_results[neighbour - 1]
                base_param_gt = gt_param_results[neighbour - 1]
            gen_dof = outputs[sample_id, neighbour, :]
            gt_dof = labels[sample_id, neighbour, :]
            gen_param = tools.get_next_pos(trans_params1=base_param_gen, dof=gen_dof,
                                           cam_cali_mat=calib_mat[sample_id, :, :])
            gt_param = tools.get_next_pos(trans_params1=base_param_gt, dof=gt_dof,
                                          cam_cali_mat=calib_mat[sample_id, :, :])
            gen_param_results.append(gen_param)
            gt_param_results.append(gt_param)
        gen_param_results = np.asarray(gen_param_results)
        gt_param_results = np.asarray(gt_param_results)
        # print('gen_param_results shape {}'.format(gen_param_results.shape))

        result_pts = tools.params2corner_pts(params=gen_param_results, cam_cali_mat=calib_mat[sample_id, :, :])
        gt_pts = tools.params2corner_pts(params=gt_param_results, cam_cali_mat=calib_mat[sample_id, :, :])
        # print(result_pts.shape, gt_pts.shape)
        # time.sleep(30)

        results_final_vec = np.mean(result_pts[-1, :, :], axis=0)
        gt_final_vec = np.mean(gt_pts[-1, :, :], axis=0)
        final_drift = np.linalg.norm(results_final_vec - gt_final_vec) * 0.2
        final_drifts.append(final_drift)
        # print(results_final_vec, gt_final_vec)
        # print(final_drift)
        # time.sleep(30)

        sample_error = tools.evaluate_dist(pts1=gt_pts, pts2=result_pts)
        batch_errors.append(sample_error)

    batch_errors = np.asarray(batch_errors)
    avg_batch_error = np.asarray(np.mean(batch_errors))

    error_tensor = torch.tensor(avg_batch_error, requires_grad=True)
    error_tensor = error_tensor.type(torch.FloatTensor)
    error_tensor = error_tensor.to(device)
    error_tensor = error_tensor * 0.99
    # print('disloss device {}'.format(device))
    # print(error_tensor)
    # time.sleep(30)

    avg_final_drift = np.asarray(np.mean(np.asarray(final_drifts)))
    final_drift_tensor = torch.tensor(avg_final_drift, requires_grad=True)
    final_drift_tensor = final_drift_tensor.type(torch.FloatTensor)
    final_drift_tensor = final_drift_tensor.to(device)
    final_drift_tensor = final_drift_tensor * 0.99
    return error_tensor, final_drift_tensor


def get_correlation_loss(labels, outputs):
    # print('labels shape {}, outputs shape {}'.format(labels.shape, outputs.shape))
    x = outputs.flatten()
    y = labels.flatten()
    # print('x shape {}, y shape {}'.format(x.shape, y.shape))
    # print('x shape\n{}\ny shape\n{}'.format(x, y))
    xy = x * y
    mean_xy = torch.mean(xy)
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    cov_xy = mean_xy - mean_x * mean_y
    # print('xy shape {}'.format(xy.shape))
    # print('xy {}'.format(xy))
    # print('mean_xy {}'.format(mean_xy))
    # print('cov_xy {}'.format(cov_xy))

    var_x = torch.sum((x - mean_x) ** 2 / x.shape[0])
    var_y = torch.sum((y - mean_y) ** 2 / y.shape[0])
    # print('var_x {}'.format(var_x))

    corr_xy = cov_xy / (torch.sqrt(var_x * var_y))
    # print('correlation_xy {}'.format(corr_xy))

    loss = 1 - corr_xy
    # time.sleep(30)
    # x = output
    # y = target
    #
    # vx = x - torch.mean(x)
    # vy = y - torch.mean(y)
    #
    # loss = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    # print('correlation loss {}'.format(loss))
    # time.sleep(30)
    return loss



#

# ----- #
def _get_random_value(r, center, hasSign):
    randNumber = random.random() * r + center


    if hasSign:
        sign = random.random() > 0.5
        if sign == False:
            randNumber *= -1

    return randNumber


# ----- #
def get_array_from_itk_matrix(itk_mat):
    mat = np.reshape(np.asarray(itk_mat), (3, 3))
    return mat


# ----- #
def create_transform(aX, aY, aZ, tX, tY, tZ, mat_base=None):
    if mat_base is None:
        mat_base = np.identity(3)

    t_all = np.asarray((tX, tY, tZ))

    # Get the transform
    rotX = sitk.VersorTransform((1, 0, 0), aX / 180.0 * np.pi)
    matX = get_array_from_itk_matrix(rotX.GetMatrix())
    #
    rotY = sitk.VersorTransform((0, 1, 0), aY / 180.0 * np.pi)
    matY = get_array_from_itk_matrix(rotY.GetMatrix())
    #
    rotZ = sitk.VersorTransform((0, 0, 1), aZ / 180.0 * np.pi)
    matZ = get_array_from_itk_matrix(rotZ.GetMatrix())

    # Apply all the rotations
    mat_all = matX.dot(matY.dot(matZ.dot(mat_base[:3, :3])))

    return mat_all, t_all


def train_model(model, criterion, optimizer, scheduler, fn_save, num_epochs=25):
    since = time.time()

    lowest_loss = 2000
    lowest_dist = 2000
    best_ep = 0
    tv_hist = {'train': [], 'val': []}
    # print('trainmodel device {}'.format(device))
    dof_label_stats = np.zeros((1, 6))

    for epoch in range(num_epochs):
        global current_epoch
        current_epoch = epoch + 1

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            # print('Network is in {}...'.format(phase))

            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_dist = 0.0
            running_dofl = 0.0
            running_corr = 0.0
            running_recon = 0.0
            # running_corrects = 0

            # Iterate over data.
            for case_id, vol_tensor, frame_tensor, mat_tensor, dof_tensor, frameid_tensor in dataloaders[phase]:
                # Get images from inputs
                # print('*'*10 + ' printing inputs and labels ' + '*'*10)

                vol_tensor = vol_tensor.type(torch.FloatTensor)
                frame_tensor = frame_tensor.type(torch.FloatTensor)
                mat_tensor = mat_tensor.type(torch.FloatTensor)
                dof_tensor = dof_tensor.type(torch.FloatTensor)
                # print('vol_tensor {}'.format(vol_tensor.shape))
                # print('frame_tensor {}'.format(frame_tensor.shape))
                # print('mat_tensor {}'.format(mat_tensor.shape))
                # print('dof_tensor {}'.format(dof_tensor.shape))
                
                # sys.exit()


                vol_tensor = vol_tensor.to(device)
                frame_tensor = frame_tensor.to(device)
                mat_tensor = mat_tensor.to(device)
                dof_tensor = dof_tensor.to(device)

                # vol_tensor.require_grad = True
                # frame_tensor.require_grad = True
                mat_tensor.require_grad = True
                dof_tensor.require_grad = True

                # zero the parameter gradients

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # print('mat_tensor\n{}'.format(mat_tensor))
                    # print('dof_tensor\n{}'.format(dof_tensor))
                    vol_resampled, dof_6 = model(vol=vol_tensor, frame=frame_tensor, device=device)
                    # print('vol_resampled {}'.format(vol_resampled.shape))
                    # print('dof_6 {}'.format(dof_6))
                    # sys.exit()

                    # dof_6 = dof_6[:, 2:4]
                    # dof_tensor = dof_tensor[:, 2:4]
                    dof_label = dof_tensor[0, :].cpu().detach().unsqueeze(0).numpy()
                    dof_pred = dof_6[0, :].cpu().detach().unsqueeze(0).numpy()
                    dof_compare = np.concatenate((dof_label, dof_pred), axis=0).astype(np.float)
                    np.set_printoptions(precision=2)

                    if normalize_dof:
                        dof_compare = dof_compare * dof_std + dof_means
                    print('{} frame {}'.format(case_id[0], frameid_tensor[0, :]))
                    print(dof_compare)

                    # dof_labels = dof_tensor.cpu().detach().numpy()
                    # dof_label_stats = np.concatenate((dof_label_stats, dof_labels), axis=0)
                    # np.savetxt('infos/label_stats.txt', dof_label_stats)

                    # sys.exit()
                    

                    
                    

                    slice_id = int(vol_tensor.shape[2] / 2)
                    indices = torch.tensor([slice_id]).to(device)
                    out_frame_tensor = torch.index_select(vol_resampled, 2, indices).squeeze()
                    # out_frame_tensor = vol_resampled[:, :, slice_id, :, :].squeeze()
                    frame_tensor = frame_tensor.squeeze()

                    # out_frame_tensor = out_frame_tensor.unsqueeze(1)
                    # frame_tensor = frame_tensor.unsqueeze(1)
                    # out_frame_tensor = out_frame_tensor.view(out_frame_tensor.size(0), -1)
                    # frame_tensor = frame_tensor.view(frame_tensor.size(0), -1)
                    # print('out {}, frame {}'.format(out_frame_tensor.shape, frame_tensor.shape))
                    # sys.exit()
                    # img_loss = criterion(out_frame_tensor, frame_tensor)
                    # img_loss = criterion_img(out_frame_tensor, frame_tensor)
                    # ssim_loss = 1 - ssim_module(out_frame_tensor, frame_tensor)
                    param_loss = criterion_mse(dof_6, dof_tensor)
                    # param_loss = criterion_l1(dof_6, dof_tensor)
                    param_cor_loss = get_correlation_loss(labels=dof_tensor, outputs=dof_6)
                    # param_loss = 0
                    # param_cor_loss = 0
                    loss = param_loss
                    # loss = param_loss + param_cor_loss * 10
                    # loss = img_loss
                    # loss = img_loss + mat_loss * 200
                    # loss = ssim_loss * 20 + mat_loss

                    mat_tensor_convert = tools.dof2mat_tensor(input_dof=dof_tensor, device=device)
                    grid = tools.myAffineGrid2(input_tensor=vol_tensor, input_mat=mat_tensor_convert, 
                                   input_spacing=(1, 1, 1), device=device)
                    grid = grid.to(device)
                    # gt_resampled = F.grid_sample(vol_tensor, grid, align_corners=True)
                    gt_resampled = F.grid_sample(vol_tensor, grid)
                    # print('gt_resampled {}'.format(gt_resampled.shape))
                    for i in range(0, 1):
                    # for i in range(vol_resampled.shape[0]):
                        # print('case_id {}'.format(case_id[i]))
                        # print(case_id[i])
                        
                        gt_np = gt_resampled[i, :, slice_id, :, :].squeeze().cpu().detach().numpy().copy()
                        # print('out_frame_tensor shape {}'.format(out_frame_tensor.shape))
                        # print('frame_tensor shape {}'.format(frame_tensor.shape))
                        out_np = out_frame_tensor.cpu().detach().squeeze().numpy()
                        frame_np = frame_tensor.cpu().squeeze().numpy()
                        # print('out_np dim {}'.format(out_np.ndim))
                        if out_np.ndim > 2:
                            out_np = out_np[i, :, :]
                            frame_np = frame_np[i, :, :]

                        cat_np = np.concatenate((frame_np, gt_np, out_np), axis=0)
                        cv2.imwrite('figures/training_img/{}.jpg'.format(case_id[i]), cat_np)
                        cv2.imwrite('tmp.jpg', cat_np)
                        # sys.exit()

                    # sys.exit()

                    """ WORKING stn sample for a batch """
                    # mat_tensor_convert = tools.dof2mat_tensor(input_dof=dof_tensor).to(device)
                    # grid = tools.myAffineGrid2(input_tensor=vol_tensor, input_mat=mat_tensor_convert, 
                    #                            input_spacing=(1, 1, 1), device=device)
                    # grid = grid.to(device)
                    # us_tensor_resampled = F.grid_sample(vol_tensor, grid, align_corners=True)
                    # resampled_array = us_tensor_resampled.squeeze().cpu().detach().numpy().copy()
                    # print('resampled_array shape {}'.format(resampled_array.shape))
                    # print('us_tensor_resampled shape {}'.format(us_tensor_resampled.shape))
                    # slice_id = 0 + int(resampled_array.shape[1] / 2)
                    # case_idx = 1
                    # stn_resampled = resampled_array[case_idx, slice_id, :, :]
                    # frame_array = frame_tensor.squeeze().cpu().numpy().copy()
                    # frame_array = frame_array[case_idx, :, :]
                    # diff_np = stn_resampled - frame_array
                    # print('max {}, min {}, mean {}'.format(np.max(diff_np), np.min(diff_np), np.mean(diff_np)))
                    # concate_np = np.concatenate((frame_array, stn_resampled, diff_np), axis=0)
                    # cv2.imwrite('tmp.jpg', concate_np)
                    # us_frame_tensor = us_tensor_resampled[:, 0, slice_id, :, :]
                    # print('resampled tensor {}'.format(us_frame_tensor.shape))
                    # print('frames tensor {}'.format(frame_tensor.shape))
                    # loss = criterion_mse(us_frame_tensor, frame_tensor)
                    # print('loss {}'.format(loss))
                    # sys.exit()
                    loss_item = float(loss.item())
                    # print(loss_item)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                        # del out_frame_tensor, vol_resampled, loss, img_loss
                        # sys.exit()
                    # print('update loss')
                    # time.sleep(30)
                # statistics
                running_loss += loss_item * vol_tensor.size(0)
                
            # print('epoch ends here')
            # sys.exit()
            epoch_loss = running_loss / dataset_sizes[phase]

            tv_hist[phase].append([float(epoch_loss), float(param_loss), float(param_cor_loss)])
            # print('tv_hist\n{}'.format(tv_hist))

            # deep copy the model
            if phase == 'val':
            # if (phase == 'val' and epoch_loss <= lowest_loss) or current_epoch % 10 == 0:
            # if phase == 'val' and epoch_loss <= lowest_loss and current_epoch > 40:
            # if phase == 'val' and epoch_dist <= lowest_dist:
                lowest_loss = epoch_loss
                # lowest_dist = epoch_dist
                best_ep = epoch
                torch.save(model.state_dict(), fn_save)
                # print('**** best model updated with dist={:.4f} ****'.format(lowest_dist))
                print('**** best model updated with loss={:.4f} ****'.format(lowest_loss))

        update_info(best_epoch=best_ep+1, current_epoch=epoch+1, lowest_val_TRE=lowest_loss)
        print('{}/{}: Tl: {:.4f}, Vl: {:.4f}'.format(
            epoch + 1, num_epochs,
            tv_hist['train'][-1][0],
            tv_hist['val'][-1][0]
        ))
        # sys.exit()
        # print('{}/{}: Tl: {:.4f}, Vl: {:.4f}, Td: {:.4f}, Vd: {:.4f}, Tc: {:.4f}, Vc: {:.4f}'.format(
        #     epoch + 1, num_epochs,
        #     tv_hist['train'][-1][0],
        #     tv_hist['val'][-1][0],
        #     tv_hist['train'][-1][1],
        #     tv_hist['val'][-1][1],
        #     tv_hist['train'][-1][2],
        #     tv_hist['val'][-1][2])
        # )
        # time.sleep(30)
        # training_progress[epoch][0] = tv_hist['train'][-1][0]
        # training_progress[epoch][1] = tv_hist['val'][-1][0]
        training_progress_new.append([tv_hist['train'][-1][0], tv_hist['val'][-1][0]])
        np.savetxt(txt_path, np.asarray(training_progress_new))
        # print('progress {}'.format(np.asarray(training_progress_new).shape))
        # print('progress {}'.format(training_progress_new))

        # fn_hist = os.path.join('data/training_loss/hist_{}.npy'.format(now_str))
        # np.save(txt_path, tv_hist)

    time_elapsed = time.time() - since
    print('*' * 10 + 'Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('*' * 10 + 'Lowest val TRE: {:4f} at epoch {}'.format(lowest_dist, best_ep))
    print()

    return tv_hist

def save_info():
    file = open('infos/experiment_diary/{}.txt'.format(now_str), 'a+')
    file.write('Time_str: {}\n'.format(now_str))
    # file.write('Initial_mode: {}\n'.format(args.init_mode))
    file.write('Training_mode: {}\n'.format(args.training_mode))
    file.write('Model_filename: {}\n'.format(args.model_filename))
    file.write('Device_no: {}\n'.format(args.device_no))
    file.write('Epochs: {}\n'.format(args.epochs))
    file.write('Network_type: {}\n'.format(args.network_type))
    file.write('Learning_rate: {}\n'.format(args.learning_rate))
    # file.write('Neighbour_slices: {}\n'.format(args.neighbour_slice))
    # file.write('Infomation: {}\n'.format(args.information))
    file.write('Best_epoch: 0\n')
    file.write('Val_loss: {:.4f}\n'.format(1000))
    file.close()
    print('Information has been saved!')

def update_info(best_epoch, current_epoch, lowest_val_TRE):
    readFile = open('data/experiment_diary/{}.txt'.format(now_str))
    lines = readFile.readlines()
    readFile.close()

    file = open('data/experiment_diary/{}.txt'.format(now_str), 'w')
    file.writelines([item for item in lines[:-2]])
    file.write('Best_epoch: {}/{}\n'.format(best_epoch, current_epoch))
    file.write('Val_loss: {:.4f}'.format(lowest_val_TRE))
    file.close()
    print('Info updated in {}!'.format(now_str))


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

    network_type = args.network_type
    print('Training mode: {}'.format(args.training_mode))

    dataset_dir = 'infos/sets/all_cases'
    # dataset_dir = 'infos/sets/spacing0.5'
    # dataset_dir = 'infos/sets/spacing0.5_tiny'
    train_ids = np.loadtxt(path.join(dataset_dir, 'train.txt'))
    val_ids = np.loadtxt(path.join(dataset_dir, 'val.txt'))
    test_ids = np.loadtxt(path.join(dataset_dir, 'test.txt'))
    image_datasets = {x: FreehandUS4D(os.path.join(dataset_dir, '{}.txt'.format(x)))
                      for x in ['train', 'val']}
    print('image_dataset\n{}'.format(image_datasets))
    # time.sleep(30)

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=0)
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    print('Number of training samples: {}'.format(dataset_sizes['train']))
    print('Number of validation samples: {}'.format(dataset_sizes['val']))

    model_folder = '/zion/guoh9/US_recon/results'
    model_path = path.join(model_folder, '3d_best_Generator_{}.pth'.format(pretrain_model_str))  # 10
    # model_ft = mynet.mynet3
    model_ft = test_network2.defineModel(model_type='mynet3_150', device=device)
    # normalize_dof = True
    normalize_dof = False

    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.SmoothL1Loss()
    criterion_img = nn.MSELoss()
    ssim_module = SSIM(data_range=255, size_average=True, channel=1)
    ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=3)

    if args.training_mode == 'finetune':
        # overwrite the learning rate for finetune
        lr = 5e-6
        print('Learning rate is overwritten to be {}'.format(lr))
    else:
        lr = args.learning_rate
        print('Learning rate = {}'.format(lr))

    optimizer = optim.Adam(model_ft.parameters(), lr=lr)
    # optimizer = optim.Adagrad(model_ft.parameters(), lr=1)
    # optimizer = optim.SGD(model_ft.parameters(), lr=lr)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    now = datetime.now()
    now_str = now.strftime('%m%d-%H%M%S')
    now_str = '150nonorm'

    save_info()

    # Train and evaluate
    fn_best_model = path.join(results_dir, '3d_best_{}_{}.pth'.format(net, now_str))

    # model_ft.load_state_dict(torch.load(fn_best_model, map_location='cuda:0'))

    print('Start training...')
    print('This model is <3d_best_{}_{}_{}.pth>'.format(net, now_str, '0'))
    txt_path = path.join(results_dir, 'training_progress_{}_{}.txt'.format(net, now_str))
    hist_ft = train_model(model_ft,
                          criterion_mse,
                          optimizer,
                          exp_lr_scheduler,
                          fn_best_model,
                          num_epochs=epochs)

    # fn_hist = os.path.join(results_dir, 'hist_{}_{}_{}.npy'.format(net, now_str, init_mode))
    # np.save(fn_hist, hist_ft)

    # np.savetxt(txt_path, training_progress)

    # now = datetime.now()
    # now_stamp = now.strftime('%Y-%m-%d %H:%M:%S')
    # print('#' * 15 + ' Training {} completed at {} '.format(init_mode, now_stamp) + '#' * 15)
