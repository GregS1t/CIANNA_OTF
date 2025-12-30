#!/usr/bin/env python

#Adrien ANTHORE, 09 Oct 2023
#Env: Python 3.6.7
#fwd_racs.py

import sys
import os
sys.path.insert(0,'/home/gsainton/Documents/01_Observatoire/CIANNA1123/src/build/lib.linux-x86_64-cpython-311')
import CIANNA as cnn

from aux_fct import *
path = "/home/gsainton/Documents/01_Observatoire/RACS_analysis"

patch_shift = 240
orig_offset = 128

max_nb_obj_per_image = 280

#Size priors for all possible boxes per grid. element
prior_w = f_ar([6.0,6.0,6.0,6.0,6.0,12.0,12.0,24.0])
prior_h = f_ar([6.0,6.0,6.0,6.0,6.0,12.0,12.0,24.0])
prior_size = np.vstack((prior_w, prior_h))

#No obj probability prior to rebalance the size distribution
prior_noobj_prob = f_ar([0.2,0.2,0.2,0.2,0.2,0.02,0.02,0.02])

#Relative scaling of each extra paramater
param_ind_scales = f_ar([2.0,2.0,1.0,0.5,0.5])

cnn.init(in_dim=i_ar([image_size,image_size]), in_nb_ch=1, out_dim=1+max_nb_obj_per_image*(7+nb_param+1),
                bias=0.1, b_size=16, comp_meth='C_CUDA', dynamic_load=1, mixed_precision="FP32C_FP32A", adv_size=35)

##### YOLO parameters tuning #####

#Relative scaling of each error "type" : 
error_scales = cnn.set_error_scales(position = 36.0, size = 0.20, probability = 0.5, objectness = 2.0, parameters = 5.0)

#Various IoU limit conditions
IoU_limits = cnn.set_IoU_limits(good_IoU_lim = 0.4, low_IoU_best_box_assoc = -1.0, min_prob_IoU_lim = -0.3,
                                                                        min_obj_IoU_lim = -0.3, min_param_IoU_lim = 0.1, diff_IoU_lim = 0.4, diff_obj_lim = 0.4)

#Activate / deactivate some parts of the loss
fit_parts = cnn.set_fit_parts(position = 1, size = 1, probability = 1, objectness = 1, parameters = 1)


slopes_and_maxes = cnn.set_slopes_and_maxes(
                                        position    = cnn.set_sm_single(slope = 0.5, fmax = 6.0, fmin = -6.0),
                                        size        = cnn.set_sm_single(slope = 0.5, fmax = 1.6, fmin = -1.4),
                                        probability = cnn.set_sm_single(slope = 0.2, fmax = 6.0, fmin = -6.0),
                                        objectness  = cnn.set_sm_single(slope = 0.5, fmax = 6.0, fmin = -6.0),
                                        parameters  = cnn.set_sm_single(slope = 0.5, fmax = 1.5, fmin = -0.2))

strict_box_size = 1

nb_yolo_filters = cnn.set_yolo_params(nb_box = nb_box, nb_class = nb_class, nb_param = nb_param, max_nb_obj_per_image = max_nb_obj_per_image,
                                prior_size = prior_size, prior_noobj_prob = prior_noobj_prob, IoU_type = "DIoU", prior_dist_type = "SIZE", 
                                error_scales = error_scales, param_ind_scales = param_ind_scales, slopes_and_maxes = slopes_and_maxes, IoU_limits = IoU_limits,
                                fit_parts = fit_parts, strict_box_size = strict_box_size, min_prior_forced_scaling = 0.0, diff_flag = 1,
                                rand_startup = nb_images*0, rand_prob_best_box_assoc = 0.0, class_softmax = 1, error_type = "natural", no_override = 1, raw_output = 1)
cnn.load("../net_save/net0_s1800.dat", 0, bin=1)

content = os.listdir(path)
list_fits = [f for f in content if os.path.isfile(os.path.join(path,f))]

for fits_file in list_fits:


	print("\nMosaic:", fits_file,"\n")

	hdul = fits.open(path+fits_file)
	full_img = hdul[0].data[0,0]
	wcs_img = WCS(hdul[0].header)

	min_pix = np.percentile(full_img, 99.4)
	max_pix = np.percentile(full_img, 99.8)

	f = open("./fwd_res/"+fits_file[:-5]+"_min_maxpix.txt", 'w')
	np.savetxt(f, [min_pix, max_pix], delimiter='\t')
	f.close()

	map_pixel_size = np.shape(full_img)[0]
	size_px = map_pixel_size
	size_py = np.shape(full_img)[1]


	nb_area_w = int((map_pixel_size-orig_offset)/patch_shift) + 1
	nb_area_h = int((map_pixel_size-orig_offset)/patch_shift) + 1

	nb_images_all = nb_area_w*nb_area_h
	targets = np.zeros((nb_images_all,1+max_nb_obj_per_image*(7+nb_param+1)), dtype="float32")

	full_data_norm = np.clip(full_img,min_pix,max_pix)
	full_data_norm = (full_data_norm - min_pix) /(max_pix-min_pix)
	full_data_norm = np.tanh(3.0*full_data_norm)

	pred_all = np.zeros((nb_images_all, image_size*image_size), dtype="float32")
	patch = np.zeros((image_size,image_size), dtype="float32")

	for i_d in range(0,nb_images_all):

		p_y = int(i_d/nb_area_w)
		p_x = int(i_d%nb_area_w)

		xmin = p_x*patch_shift - orig_offset
		xmax = p_x*patch_shift + image_size - orig_offset
		ymin = p_y*patch_shift - orig_offset
		ymax = p_y*patch_shift + image_size - orig_offset

		px_min = 0; px_max = image_size
		py_min = 0; py_max = image_size

		set_zero = 0

		if(xmin < 0):
			px_min = -xmin
			xmin = 0
			set_zero = 1
		if(ymin < 0):
			py_min = -ymin
			ymin = 0
			set_zero = 1
		if(xmax > size_px):
			px_max = image_size - (xmax-size_px)
			xmax = size_px
			set_zero = 1
		if(ymax > size_py):
			py_max = image_size - (ymax-size_py)
			ymax = size_py
			set_zero = 1

		if(set_zero):
			patch[:,:] = 0.0

		patch[px_min:px_max,py_min:py_max] = np.flip(full_data_norm[xmin:xmax,ymin:ymax],axis=0)

		pred_all[i_d,:] = patch.flatten("C")

	input_data = pred_all

	#Forward
	cnn.create_dataset("TEST", nb_images_all, input_data[:,:], targets[:,:])
	cnn.forward(repeat=1, no_error=1, saving=2, drop_mode="AVG_MODEL")

	os.rename("./fwd_res/net0_0000.dat", "./fwd_res/net0_RACS_"+fits_file[9:-5]+".dat")
