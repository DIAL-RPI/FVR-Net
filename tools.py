# Demo image registration using SimpleITK

from matplotlib import pyplot as plt
import numpy as np
import SimpleITK as sitk
import time
import pandas as pd
from os import path
import os
import sys
import cv2
import imageio
import torch
import torchgeometry as tgm
import math
from utils import transformations as tfms
import random


uronav_dataset = '/zion/common/data/uronav_data'
usrecon_dataset = '/zion/guoh9/US_recon/US_dataset'
myvol_dataset = '/zion/guoh9/US_recon/recon'
seq_dataset = '/zion/guoh9/US_recon/new_data'


def pic2gif(folder):
    gifs = []
    for i in range(fixedArray.shape[0]):
        gifs.append(fixedArray[i, :, :])
    imageio.mimsave('plots/compare.gif', gifs, duration=0.2)


def folder2imglist(folder):
    file_list = os.listdir(folder)
    file_list.sort()
    img_list = []
    for filename in file_list:
        img_path = path.join(folder, filename)
        img_list.append(cv2.imread(img_path, 1))
    return img_list

def mat2tfm(input_mat):
    tfm = sitk.AffineTransform(3)
    tfm.SetMatrix(np.reshape(input_mat[:3, :3], (9,)))
    translation = input_mat[:3,3]
    tfm.SetTranslation(translation)
    # tfm.SetCenter([0, 0, 0])
    return tfm

def case2gif(case_id):
    multimodal_folder = 'results/{}/multimodal'.format(case_id)
    img_list = folder2imglist(folder=multimodal_folder)
    gif_path = 'results/{}/{}_fused.gif'.format(case_id, case_id)
    imageio.mimsave(gif_path, img_list, duration=0.2)
    print('{} gif saved!'.format(case_id))


def volCompare(case_id):
    uronav_case_folder = path.join(uronav_dataset, case_id)
    myvol_case_folder = path.join(myvol_dataset, case_id)
    print(os.listdir(uronav_case_folder))
    print(os.listdir(myvol_case_folder))

    vol_uronav = sitk.ReadImage(path.join(uronav_case_folder, 'USVol.mhd'),
                                sitk.sitkFloat64)
    vol_my = sitk.ReadImage(path.join(myvol_case_folder, '{}_myrecon.mhd'.format(case_id)),
                            sitk.sitkFloat64)
    print('vol_uronav\n{}'.format(vol_uronav.GetSize()))
    print('vol_my\n{}'.format(vol_my.GetSize()))

    vol_uronav_np = sitk.GetArrayFromImage(vol_uronav)
    vol_my_np = sitk.GetArrayFromImage(vol_my)

    print('uronav_np {}, my_np {}'.format(
        vol_uronav_np.shape, vol_my_np.shape))
    cv2.imwrite('tmp.jpg', vol_uronav_np[20, :, :])
    cv2.imwrite('tmp2.jpg', vol_my_np[20, :, :])


def readMatsFromSequence(case_id, type='adjusted', model_str='gt', on_arc=False):
    """ Read a sequence .mhd file and return frame_num*4*4 transformation mats

    Args:
        case_id (str): case ID like "Case0005"
        type (str, optional): Whether bottom centerline is adjuested 
        or origin. Defaults to 'adjusted'.
        model_str (str, optional): Could be model's time string. Defaults to 'gt'.

    Returns:
        Numpy array: frame_num x 4 x 4 transformation mats for each frame
    """
    if on_arc:
        case_seq_folder = '/raid/shared/guoh9/US_recon/new_data/{}'.format(case_id)
        # case_seq_folder = '/raid/shared/guoh9/US_recon'
    else:
        case_seq_folder = path.join(seq_dataset, case_id)
    # print(os.listdir(case_seq_folder))
    # sys.exit()
    case_seq_path = path.join(
        case_seq_folder, '{}_{}_{}.mhd'.format(case_id, type, model_str))

    file = open(case_seq_path, 'r')
    lines = file.readlines()
    mats = []
    for line in lines:
        words = line.split(' ')
        if words[0].endswith('ImageToProbeTransform'):
            # print(words)
            words[-1] = words[-1][:-2]
            nums = np.asarray(words[2:]).astype(np.float)
            nums.shape = (4, 4)
            mats.append(nums)
    mats = np.asarray(mats)
    return mats

def computeScale(input_mat):
    scale1 = np.linalg.norm(input_mat[:3, 0])
    scale2 = np.linalg.norm(input_mat[:3, 1])
    scale3 = np.linalg.norm(input_mat[:3, 2])
    # print('scale1 {}'.format(scale1))
    # print('scale2 {}'.format(scale2))
    # print('scale3 {}'.format(scale3))
    # print(0.478425 * 0.35)
    # sys.exit()
    return np.asarray([scale1, scale2, scale3])



def samplePlane(case_id, trans_mats, frame_id):
    us_path = path.join(myvol_dataset, '{}/{}_myrecon.mhd'.format(case_id, case_id))
    us_img = sitk.ReadImage(us_path)
    us_np = sitk.GetArrayFromImage(us_img)
    print(us_img.GetOrigin())
    print('us_np shape {}'.format(us_np.shape))
    print('us_img size {}'.format(us_img.GetSize()))
    fixed_path = path.join(usrecon_dataset, '{}/frames/{:04}.jpg'.format(case_id, frame_id))
    fixed_origin = cv2.imread(fixed_path, 0)

    clip_x, clip_y, clip_h, clip_w = 105, 54, 320, 565
    fixed_np = fixed_origin[clip_x:clip_x+clip_h, clip_y:clip_y+clip_w]
    # fixed_np = fixed_origin[105:105+320, 54:54+565]

    # spacing = 0.4   # For my Slicer reconstructed volume
    # spacing = 0.35  # For uronac reconstructed volume
    mat_scales = computeScale(input_mat=trans_mats[frame_id, :, :])
    spacing = np.mean(mat_scales[:2]) / us_img.GetSpacing()[0]
    print('frame_scale = {}'.format(spacing))
    frame_w = int(spacing * fixed_np.shape[1])
    frame_h = int(spacing * fixed_np.shape[0])
    fixed_np = cv2.resize(fixed_np, (frame_w, frame_h))
    fixed_np = fixed_np.astype(np.float64)
    fixed_np = np.expand_dims(fixed_np, axis=0)
    print('fixed_np shape {}'.format(fixed_np.shape))

    fixed_image = sitk.GetImageFromArray(fixed_np)
    # fixed_image.SetSpacing(us_img.GetSpacing())

    frame_mat = trans_mats[frame_id, :, :]
    # print('us_img {}'.format(us_img))
    # print('frame_mat\n{}'.format(frame_mat))


    # tfm2us = sitk.Transform(mat2tfm(np.identity(4)))
    # affine_tfm = sitk.AffineTransform(3)
    # affine_tfm.SetMatrix(frame_mat[:3, :3].flatten())
    # affine_tfm.SetTranslation(frame_mat[:3, 3])
    # print(affine_tfm)

    # spacing1 = us_img.GetSpacing()[0]
    # print('spacing1 {}, spacing {}'.format(spacing1, spacing))
    # width, length = fixed_origin.shape[1], fixed_origin.shape[0]
    destVol = sitk.Image(int(clip_w*spacing), int(clip_h*spacing), 1, sitk.sitkUInt8)
    destSpacing = np.asarray([spacing, spacing, spacing])
    destVol.SetSpacing((1/destSpacing[0], 1/destSpacing[1], 1/destSpacing[2]))
    corner = np.asarray([clip_y, clip_x, 0])
    trans_corner = sitk.TranslationTransform(3, corner.astype(np.float64))

    # computeScale(input_mat=frame_mat)

    # tfm2us = sitk.Transform(mat2tfm(np.identity(4)))
    tfm2us = sitk.Transform(mat2tfm(input_mat=frame_mat))
    tfm2us.AddTransform(trans_corner)
    print(tfm2us)

    """ US volume resampler, with final_transform"""
    resampler_us = sitk.ResampleImageFilter()
    resampler_us.SetReferenceImage(destVol)
    resampler_us.SetInterpolator(sitk.sitkLinear)
    resampler_us.SetDefaultPixelValue(0)
    resampler_us.SetTransform(tfm2us)
    outUSImg = resampler_us.Execute(us_img)
    outUSNp = sitk.GetArrayFromImage(outUSImg[:, :, 0])
    print('outUSNp shape {}'.format(outUSNp.shape))

    resampler_slice = sitk.ResampleImageFilter()
    resampler_slice.SetReferenceImage(destVol)
    resampler_slice.SetInterpolator(sitk.sitkLinear)
    resampler_slice.SetDefaultPixelValue(0)
    resampler_slice.SetTransform(trans_corner)
    outFrameImg = resampler_slice.Execute(sitk.GetImageFromArray(np.expand_dims(fixed_origin, axis=0)))
    # outFrameImg = resampler_slice.Execute(fixed_image)
    outFrameNp = sitk.GetArrayFromImage(outFrameImg[:, :, 0])
    print('fixed_origin shape {}'.format(outFrameNp.shape))

    frame_resample_concate = np.concatenate((outFrameNp, outUSNp), axis=0)
    cv2.imwrite('tmp.jpg', frame_resample_concate)

def cell_images():
    set_path = '/home/guoh9/tmp/cells/full_frames'
    case_id_list = os.listdir(set_path)
    print(os.listdir(set_path))

    for i in range(1, 33):
        case_id = 'XY{:02}_video'.format(i)
        frame0_path = path.join(set_path, case_id, 'frame0.jpg')
        print(frame0_path)
        frame0 = cv2.imread(frame0_path, 0)
        target_path = path.join(set_path, 'collections/{}.jpg'.format(case_id))
        cv2.imwrite(target_path, frame0)
        print('{} frame0 saved'.format(case_id))

def myAffineGrid(input_tensor, input_mat, input_spacing=[1, 1, 1]):
    input_spacing = np.asarray(input_spacing)
    image_size = np.asarray([input_tensor.shape[4], input_tensor.shape[3], input_tensor.shape[2]])
    image_phy_size = (image_size - 1) * input_spacing
    # image_phy_size = [input_tensor.shape[4], input_tensor.shape[3], input_tensor.shape[2]]
    grid_size = input_tensor.shape
    t_mat = input_mat
    image_tensor = input_tensor

    # generate grid of input image (i.e., the coordinate of the each pixel in the input image. The center point of the input image volume is assigned as (0, 0, 0).)
    grid_x_1d = torch.linspace(-0.5 * image_phy_size[0], 0.5 * image_phy_size[0], steps=grid_size[4])
    grid_y_1d = torch.linspace(-0.5 * image_phy_size[1], 0.5 * image_phy_size[1], steps=grid_size[3])
    grid_z_1d = torch.linspace(-0.5 * image_phy_size[2], 0.5 * image_phy_size[2], steps=grid_size[2])
    grid_z, grid_y, grid_x = torch.meshgrid(grid_z_1d, grid_y_1d, grid_x_1d)
    grid_x = grid_x.unsqueeze(0)
    grid_y = grid_y.unsqueeze(0)
    grid_z = grid_z.unsqueeze(0)
    origin_grid = torch.cat([grid_x, grid_y, grid_z, torch.ones_like(grid_x)], dim=0)
    origin_grid = origin_grid.view(4, -1)

    # compute the rasample grid through matrix multiplication
    print('t_mat {}, origin_grid {}'.format(t_mat.shape, origin_grid.shape))
    print('img_tensor type {}'.format(image_tensor.type()))
    t_mat = torch.tensor(t_mat)
    t_mat = t_mat.float()
    # origin_grid = origin_grid.unsqueeze(0)
    print('t_mat shape {}'.format(t_mat.shape))
    print('origin_grid shape {}'.format(origin_grid.shape))
    resample_grid = torch.matmul(t_mat, origin_grid)[0:3, :]

    # convert the resample grid coordinate from physical coordinate system to a range of [-1, 1] (which is required by the PyTorch interface 'grid_sample'). 
    resample_grid[0, :] = (resample_grid[0, :] + 0.5 * image_phy_size[0]) / image_phy_size[0] * 2 - 1
    resample_grid[1, :] = (resample_grid[1, :] + 0.5 * image_phy_size[1]) / image_phy_size[1] * 2 - 1
    resample_grid[2, :] = (resample_grid[2, :] + 0.5 * image_phy_size[2]) / image_phy_size[2] * 2 - 1
    print('before {}'.format(resample_grid.shape))
    resample_grid = resample_grid.permute(1,0)
    print('after {}'.format(resample_grid.shape))
    resample_grid = resample_grid.contiguous()
    print('after2 {}'.format(resample_grid.shape))
    resample_grid = resample_grid.reshape(grid_size[2], grid_size[3], grid_size[4], 3)
    resample_grid = resample_grid.unsqueeze(0)
    print('resample_grid {}'.format(resample_grid.shape))
    # sys.exit()
    return resample_grid.double()

def myAffineGrid2(input_tensor, input_mat, input_spacing=[1, 1, 1], device=None):
    # print('input_tensor shape {}'.format(input_tensor.shape))
    # print('input_mat shape {}'.format(input_mat.shape))
    # sys.exit()

    input_spacing = np.asarray(input_spacing)
    image_size = np.asarray([input_tensor.shape[4], input_tensor.shape[3], input_tensor.shape[2]])
    image_phy_size = (image_size - 1) * input_spacing
    # image_phy_size = [input_tensor.shape[4], input_tensor.shape[3], input_tensor.shape[2]]
    grid_size = input_tensor.shape

    # generate grid of input image (i.e., the coordinate of the each pixel in the input image. The center point of the input image volume is assigned as (0, 0, 0).)
    grid_x_1d = torch.linspace(-0.5 * image_phy_size[0], 0.5 * image_phy_size[0], steps=grid_size[4])
    grid_y_1d = torch.linspace(-0.5 * image_phy_size[1], 0.5 * image_phy_size[1], steps=grid_size[3])
    grid_z_1d = torch.linspace(-0.5 * image_phy_size[2], 0.5 * image_phy_size[2], steps=grid_size[2])
    grid_z, grid_y, grid_x = torch.meshgrid(grid_z_1d, grid_y_1d, grid_x_1d)
    grid_x = grid_x.unsqueeze(0)
    grid_y = grid_y.unsqueeze(0)
    grid_z = grid_z.unsqueeze(0)
    origin_grid = torch.cat([grid_x, grid_y, grid_z, torch.ones_like(grid_x)], dim=0)
    origin_grid = origin_grid.view(4, -1)
    if device:
        origin_grid = origin_grid.to(device)
        origin_grid.requires_grad = True

    # compute the rasample grid through matrix multiplication
    # print('t_mat {}, origin_grid {}'.format(t_mat.shape, origin_grid.shape))
    # t_mat = input_mat
    # t_mat = torch.tensor(t_mat)
    # t_mat = t_mat.float()
    # t_mat.requires_grad = True

    # t_mat = t_mat.squeeze()
    # origin_grid = origin_grid.unsqueeze(0)
    # print('t_mat shape {}'.format(t_mat.shape))
    # print('origin_grid shape {}'.format(origin_grid.shape))
    # resample_grid = torch.matmul(t_mat, origin_grid)[0:3, :]
    resample_grid = torch.matmul(input_mat, origin_grid)[:, 0:3, :]
    # print('resample_grid {}'.format(resample_grid.shape))

    # convert the resample grid coordinate from physical coordinate system to a range of [-1, 1] (which is required by the PyTorch interface 'grid_sample'). 
    resample_grid[:, 0, :] = (resample_grid[:, 0, :] + 0.5 * image_phy_size[0]) / image_phy_size[0] * 2 - 1
    resample_grid[:, 1, :] = (resample_grid[:, 1, :] + 0.5 * image_phy_size[1]) / image_phy_size[1] * 2 - 1
    resample_grid[:, 2, :] = (resample_grid[:, 2, :] + 0.5 * image_phy_size[2]) / image_phy_size[2] * 2 - 1
    # print('resample_grid2 {}'.format(resample_grid.shape))
    resample_grid = resample_grid.permute(0,2,1).contiguous()
    resample_grid = resample_grid.reshape(grid_size[0], grid_size[2], grid_size[3], grid_size[4], 3)
    # resample_grid = resample_grid.unsqueeze(1)
    # print('resample_grid {}'.format(resample_grid.shape))
    # sys.exit()
    return resample_grid

def processFrame(us_spacing, frame_np, frame_mat, clip_info):
    """Crop the frame with reconstruction ROI, respacing to the same as US volume

    Args:
        us_spacing (tuple): sitk_img.GetSpacing()
        frame_np (np array): Raw 1-channel grey image from frame
        frame_mat ([np array]): 4x4 matrix of this frame, read from sequence mhd file

    Returns:
        fixed_np: cropped and resize frame ROI
    """
    # print('us_spacing {}'.format(us_spacing))
    # print('frame_np {}'.format(frame_np))
    # print('frame_mat {}'.format(frame_mat))
    # print('clip_info {}'.format(clip_info))
    # sys.exit()
    clip_x, clip_y, clip_h, clip_w = clip_info

    fixed_np = frame_np[clip_x:clip_x+clip_h, clip_y:clip_y+clip_w]
    mat_scales = computeScale(input_mat=frame_mat)
    # print('matscales {}'.format(mat_scales))
    spacing = np.mean(mat_scales[:2]) / us_spacing[0]
    frame_w = int(spacing * fixed_np.shape[1])
    frame_h = int(spacing * fixed_np.shape[0])
    fixed_np = cv2.resize(fixed_np, (frame_w, frame_h))
    fixed_np = fixed_np.astype(np.float64)
    return fixed_np

def mat2dof_np(input_mat):
    # print('input_mat\n{}'.format(input_mat))
    translations = input_mat[:3, 3]
    rotations_eulers = np.asarray(tfms.euler_from_matrix(input_mat))
    rotations_degrees = (rotations_eulers / (2 * math.pi)) * 360
    scales = computeScale(input_mat=input_mat)

    dof = np.concatenate((translations, rotations_degrees, scales), axis=0)

    # print('dof\n{}\n'.format(dof))
    # sys.exit()
    return dof

def dof2mat_np(input_dof, scale=False):
    """ Transfer degrees to euler """
    dof = input_dof
    # print('deg {}'.format(dof[3:6]))
    dof[3:6] = dof[3:6] * (2 * math.pi) / 360.0
    # print('rad {}'.format(dof[3:6]))


    rot_mat = tfms.euler_matrix(dof[5], dof[4], dof[3], 'rzyx')[:3, :3]

    mat44 = np.identity(4)
    mat44[:3, :3] = rot_mat
    mat44[:3, 3] = dof[:3]

    if scale:
        scales = dof[6:]
        mat_scale = np.diag([scales[1], scales[0], scales[2], 1])
        mat44 = np.dot(mat44, np.linalg.inv(mat_scale))
    # print('mat_scale\n{}'.format(mat_scale))
    # print('recon mat\n{}'.format(mat44))
    # sys.exit()
    return mat44

def matSitk2Stn(input_mat, clip_size, raw_spacing, frame_shape,
                img_size, img_spacing, img_origin):
    frame_gt_mat = input_mat
    clip_x, clip_y = clip_size
    corner = np.asarray([clip_y, clip_x, 0])
    
    pos_spacing = np.mean(computeScale(input_mat=frame_gt_mat))
    spacing_mat = np.diag([1/pos_spacing, 1/pos_spacing, 1/pos_spacing, 1])
    trans_mat = np.identity(4)
    trans_mat[:3, 3] = corner
    frame_gt_mat[:3, 3] -= img_origin
    frame_gt_mat = np.dot(frame_gt_mat, trans_mat)
    frame_gt_mat = np.dot(frame_gt_mat, spacing_mat)
    frame_gt_mat[:3, 3] *= [img_spacing[0]/raw_spacing[0],
                            img_spacing[1]/raw_spacing[1], 
                            img_spacing[2]/raw_spacing[2]]

    """ origin_translate makes the volume center at coordinate center """
    origin_translate = np.identity(4)
    origin_translate[:3, 3] = -0.5 * np.asarray(img_size) * np.asarray(img_spacing)

    """ dest_translate makes the resultant sampling plane at the coordinate center"""
    dest_translate = np.identity(4)
    dest_translate[:3, 3] = np.asarray([frame_shape[1]/2, frame_shape[0]/2,0])

    frame_gt_mat = np.dot(origin_translate, frame_gt_mat)
    frame_gt_mat = np.dot(frame_gt_mat, dest_translate)
    return frame_gt_mat

def volContainer(input_tensor, container_size=(292, 158, 229)):
    # print('input_tensor shape {}'.format(input_tensor.shape))
    input_shape = list(input_tensor.shape)
    input_tensor_compact = torch.squeeze(input_tensor)
    vol_d, vol_h, vol_w = input_tensor_compact.shape
    con_d, con_h, con_w = container_size
    d_start = int((con_d-vol_d)/2)
    h_start = int((con_h-vol_h)/2)
    w_start = int((con_w-vol_w)/2)
    # print('vol_d {}, vol_h {}, vol_w {}'.format(vol_d, vol_h, vol_w))
    # print('d_start {}, h_start {}, w_start {}'.format(d_start, h_start, w_start))
    output_shape = [con_d, con_h, con_w]
    output_tensor = torch.zeros(output_shape)
    output_tensor[d_start:d_start+vol_d, h_start:h_start+vol_h, w_start:w_start+vol_w] = input_tensor_compact
    for i in range(len(input_shape)-3):
        output_tensor = output_tensor.unsqueeze(0)
    # print('output tensor shape {}'.format(output_tensor.shape))
    return output_tensor
    # sys.exit()

def frameContainer(input_tensor, container_size=(292, 158, 229), start=(0, 0)):
    # print('input_tensor shape {}'.format(input_tensor.shape))
    input_shape = list(input_tensor.shape)
    input_tensor_compact = torch.squeeze(input_tensor)
    frame_h, frame_w = input_tensor_compact.shape
    con_d, con_h, con_w = container_size
    # print('frame_h {}, frame_w {}'.format(frame_h, frame_w))
    # print('con_h {}, con_w {}'.format(con_h, con_w))
    h_start, w_start = start
    # print('vol_d {}, vol_h {}, vol_w {}'.format(vol_d, vol_h, vol_w))
    # print('h_start {}, w_start {}'.format(h_start, w_start))
    output_shape = [con_h, con_w]
    output_tensor = torch.zeros(output_shape)
    output_tensor[h_start:h_start+frame_h, w_start:w_start+frame_w] = input_tensor_compact
    for i in range(len(input_shape)-3):
        output_tensor = output_tensor.unsqueeze(0)
    # print('output tensor shape {}'.format(output_tensor.shape))
    return output_tensor

def frameCrop(input_np, crop_size=(128, 128)):
    input_h, input_w = input_np.shape
    crop_h, crop_w = crop_size
    max_h = max(input_h, crop_h)
    max_w = max(input_w, crop_w)

    if crop_h > input_h or crop_w > input_w:
        container = np.zeros((max_h, max_w))
        con_start_h = int((max_h - input_h)/2)
        con_start_w = int((max_w - input_w)/2)
        container[con_start_h:con_start_h+input_h, con_start_w:con_start_w+input_w] = input_np
        input_np = container

    start_h = int((input_np.shape[0] - crop_h)/2)
    start_w = int((input_np.shape[1] - crop_w)/2)
    output_np = input_np[start_h:start_h+crop_h, start_w:start_w+crop_w]
    return output_np

def chooseRandInit(frame_num, frame_id, rand_range=20):
    """Choose a random slice in a range [-20, 20], for subvolume initialization

    Args:
        frame_num ([int]): total number of frame
        frame_id ([int]): current frame id
        rand_range (int, optional): Range of initialization. Defaults to 20.

    Returns:
        [int]: initialization frame id
    """
    # print('num {}, id {}'.format(frame_num, frame_id))
    upper = frame_id + rand_range
    lower = frame_id - rand_range
    upper = min(upper, frame_num-1)
    lower = max(lower, 0)
    rand_id = random.randint(lower, upper)
    # print('upper {}, lower {}'.format(upper, lower))
    # print('rand_id {}'.format(rand_id))
    return rand_id

def sampleSubvol(sitk_img, init_mat, crop_size):
    # print('sitk_img origin {}'.format(sitk_img.GetOrigin()))

    source_img = sitk_img
    init_tfm = mat2tfm(input_mat=init_mat)
    # destVol = sitk.Image(sitk_img.GetSize()[0], sitk_img.GetSize()[1], 1, sitk.sitkUInt8)
    destVol = sitk.Image(crop_size[0], crop_size[1], crop_size[2], sitk.sitkUInt8)
    destSpacing = np.asarray(sitk_img.GetSpacing())
    destVol.SetSpacing((destSpacing[0], destSpacing[1], destSpacing[2]))

    destVol.SetOrigin(-0.5*np.asarray(destVol.GetSize())
                      *np.asarray(destVol.GetSpacing()))
    source_img.SetOrigin(-0.5*np.asarray(source_img.GetSize())
                                *np.asarray(source_img.GetSpacing()))
    # print('source_img origin {}'.format(source_img.GetOrigin()))
    # print('destVol origin {}'.format(destVol.GetOrigin()))
    """ US volume resampler, with frame position groundtruth """
    resampler_us = sitk.ResampleImageFilter()
    resampler_us.SetReferenceImage(destVol)
    resampler_us.SetInterpolator(sitk.sitkLinear)
    resampler_us.SetDefaultPixelValue(0)
    resampler_us.SetTransform(init_tfm)
    outUSImg = resampler_us.Execute(source_img)
    outUSNp = sitk.GetArrayFromImage(outUSImg)
    # print('outUSNp {}'.format(outUSNp.shape))
    # cv2.imwrite('tmp_sitk.jpg', outUSNp[32, :, :])
    # sys.exit()
    return outUSNp

def dof2mat_tensor(input_dof, device):
    rad = tgm.deg2rad(input_dof[:, 3:])

    ai = rad[:, 0]
    aj = rad[:, 1]
    ak = rad[:, 2]

    si, sj, sk = torch.sin(ai), torch.sin(aj), torch.sin(ak)
    ci, cj, ck = torch.cos(ai), torch.cos(aj), torch.cos(ak)
    cc, cs = ci*ck, ci*sk
    sc, ss = si*ck, si*sk

    M = torch.zeros((input_dof.shape[0], 4, 4))

    if device:
        M = M.to(device)
        M.requires_grad = True

    M[:, 0, 0] = cj*ck
    M[:, 0, 1] = sj*sc-cs
    M[:, 0, 2] = sj*cc+ss
    M[:, 1, 0] = cj*sk
    M[:, 1, 1] = sj*ss+cc
    M[:, 1, 2] = sj*cs-sc
    M[:, 2, 0] = -sj
    M[:, 2, 1] = cj*si
    M[:, 2, 2] = cj*ci
    M[:, :3, 3] = input_dof[:, :3]

    # print('out_mat {}\n{}'.format(M.shape, M))
    # sys.exit()
    return M

def computeError(mat_error, spacing, img_size):
    """[summary]

    Args:
        mat_error ([numpy]): 4x4 numpy mat, difference mat between GT and Prediction
        spacing ([float]): spacing of original usvolume
        img_size ([tuple 2]): tuple of numpy frame size, for defining corner pts

    Returns:
        [float]: error in mm
    """
    # print('mat_error\n{}'.format(mat_error))
    # print('spacing\n{}'.format(spacing))
    # print('img_size\n{}'.format(img_size))

    h, w = img_size
    corner_pts = []
    for x in [-h/2, h/2]:
        for y in [-w/2, w/2]:
            corner_pts.append([x, y, 0, 1])
    corner_pts = np.asarray(corner_pts)
    corner_pts = np.transpose(corner_pts)
    # print('corner_pts\n{}'.format(corner_pts))

    trans_corner_pts = np.dot(mat_error, corner_pts)
    # print('trans_corner_pts\n{}'.format(trans_corner_pts))

    dist = np.linalg.norm(corner_pts - trans_corner_pts, axis=0)
    # print('dist\n{}'.format(dist))

    error_mm = spacing * np.mean(dist)
    # print('error {} mm'.format(error_mm))
    
    # sys.exit()
    return error_mm

def correlation_coefficient(T1, T2):
    numerator = np.mean((T1 - T1.mean()) * (T2 - T2.mean()))
    denominator = T1.std() * T2.std()
    if denominator == 0:
        return 0
    else:
        result = numerator / denominator
        return result

def generateRandomGuess(means, stds):
    random_dof = []

    for i in range(means.shape[0]):
        this_mean, this_std = means[i], stds[i]
        rand_dof = np.random.normal(this_mean, this_std, 1)[0]
        # print('mean {:.4f}, std {:.4f}, rand {:.4f}'.format(this_mean, this_std, rand_dof))
        random_dof.append(rand_dof)
    # print(random_dof)
    # sys.exit()
    return np.asarray(random_dof)
# mats = readMatsFromSequence(case_id='Case0005')
# samplePlane(case_id='Case0005', trans_mats=mats, frame_id=43)
# print('mats shape {}'.format(mats.shape))

# volCompare(case_id='Case0009')


