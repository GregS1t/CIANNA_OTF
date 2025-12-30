
import numpy as np


from numba import jit

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import ScalarFormatter

class ScalarFormatterForceFormat(ScalarFormatter):
	def _set_format(self):  # Override function that finds format to use.
		self.format = "%1.1f"  # Give format here

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
	if isinstance(cmap, str):
		cmap = plt.get_cmap(cmap)
	new_cmap = colors.LinearSegmentedColormap.from_list(
		'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
		cmap(np.linspace(minval, maxval, n)))
	return new_cmap

plt.rcParams.update({'font.size': 10})


global map_pixel_size, beam_size, pixel_size, to_sigma_conv
global full_img, wcs_img, data_beam, wcs_beam
global image_size, im_depth, min_pix, max_pix
global nb_images, add_noise_prop, nb_valid, max_nb_obj_per_image, box_prior_class
global nb_param, nb_box, nb_class, c_size, yolo_nb_reg
global flip_hor, flip_vert, rotate_flag
global patch_shift, orig_offset, nb_area_w, nb_area_h, nb_images_all, overlap


def i_ar(int_list):
	return np.array(int_list, dtype="int")


def f_ar(float_list):
	return np.array(float_list, dtype="float32")


@jit(nopython=True, cache=True, fastmath=False)
def ellipses_to_boxes(n_sources, bmaj, bmin, angle, n_w, n_h):
	#The provided bmaj and bmin are the FWHM of a gaussian fitting of the sources
	#We use twice these dimensions to define our source sizes
	for i in range(0,n_sources):
		B = bmaj[i]
		b = bmin[i]
		alpha = angle[i]
		
		x_B = 0.5*B*np.cos(alpha*np.pi/180.0)
		y_B = 0.5*B*np.sin(alpha*np.pi/180.0)

		x_b = 0.5*b*np.cos((90.0+alpha)*np.pi/180.0)
		y_b = 0.5*b*np.sin((90.0+alpha)*np.pi/180.0)

		x_box = np.sqrt(x_B**2+x_b**2)
		y_box = np.sqrt(y_B**2+y_b**2)

		n_w[i] = 2.0*x_box
		n_h[i] = 2.0*y_box


@jit(nopython=True, cache=True, fastmath=False)
def fct_IoU(box1, box2):
	inter_w = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
	inter_h = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
	inter_2d = inter_w*inter_h
	uni_2d = (box1[2]-box1[0])*(box1[3] - box1[1]) + \
		(box2[2]-box2[0])*(box2[3] - box2[1]) - inter_2d

	return float(inter_2d)/float(uni_2d)


@jit(nopython=True, cache=True, fastmath=False)
def fct_GIoU(box1, box2):
	inter_w = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
	inter_h = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
	inter_2d = inter_w*inter_h
	uni_2d = abs(box1[2]-box1[0])*abs(box1[3] - box1[1]) + \
		abs(box2[2]-box2[0])*abs(box2[3] - box2[1]) - inter_2d
	enclose_w = (max(box1[2], box2[2]) - min(box1[0], box2[0]))
	enclose_h = (max(box1[3], box2[3]) - min(box1[1],box2[1]))
	enclose_2d = enclose_w*enclose_h

	cx_a = (box1[2] + box1[0])*0.5; cx_b = (box2[2] + box2[0])*0.5
	cy_a = (box1[3] + box1[1])*0.5; cy_b = (box2[3] + box2[1])*0.5
	dist_cent = np.sqrt((cx_a - cx_b)*(cx_a - cx_b) + (cy_a - cy_b)*(cy_a - cy_b))
	diag_enclose = np.sqrt(enclose_w*enclose_w + enclose_h*enclose_h)

	return float(inter_2d)/float(uni_2d) - float(enclose_2d - uni_2d)/float(enclose_2d)


@jit(nopython=True, cache=True, fastmath=False)
def fct_DIoU(box1, box2):
	inter_w = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
	inter_h = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
	inter_2d = inter_w*inter_h
	uni_2d = abs(box1[2]-box1[0])*abs(box1[3] - box1[1]) + \
		abs(box2[2]-box2[0])*abs(box2[3] - box2[1]) - inter_2d
	enclose_w = (max(box1[2], box2[2]) - min(box1[0], box2[0]))
	enclose_h = (max(box1[3], box2[3]) - min(box1[1],box2[1]))
	enclose_2d = enclose_w*enclose_h

	cx_a = (box1[2] + box1[0])*0.5; cx_b = (box2[2] + box2[0])*0.5
	cy_a = (box1[3] + box1[1])*0.5; cy_b = (box2[3] + box2[1])*0.5
	dist_cent = np.sqrt((cx_a - cx_b)*(cx_a - cx_b) + (cy_a - cy_b)*(cy_a - cy_b))
	diag_enclose = np.sqrt(enclose_w*enclose_w + enclose_h*enclose_h)

	return float(inter_2d)/float(uni_2d) - float(dist_cent)/float(diag_enclose)
 
@jit(nopython=True, cache=True, fastmath=False)
def fct_DIoU2(box1, box2):
	inter_w = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
	inter_h = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
	inter_2d = inter_w*inter_h
	uni_2d = abs(box1[2]-box1[0])*abs(box1[3] - box1[1]) + \
		abs(box2[2]-box2[0])*abs(box2[3] - box2[1]) - inter_2d
	enclose_w = (max(box1[2], box2[2]) - min(box1[0], box2[0]))
	enclose_h = (max(box1[3], box2[3]) - min(box1[1],box2[1]))
	enclose_2d = enclose_w*enclose_h

	cx_a = (box1[2] + box1[0])*0.5; cx_b = (box2[2] + box2[0])*0.5
	cy_a = (box1[3] + box1[1])*0.5; cy_b = (box2[3] + box2[1])*0.5
	dist_cent = ((cx_a - cx_b)*(cx_a - cx_b) + (cy_a - cy_b)*(cy_a - cy_b))
	diag_enclose = (enclose_w*enclose_w + enclose_h*enclose_h)

	return float(inter_2d)/float(uni_2d) - float(dist_cent)/float(diag_enclose)


@jit(nopython=True, cache=True, fastmath=False)
def global_to_tile_coord (offset_tab, tile_coords, priors, c_size):
	bx = (offset_tab[0] + tile_coords[1])*c_size
	by = (offset_tab[1] + tile_coords[0])*c_size
	bw = max(5.0,priors[0]*np.exp(offset_tab[3]))
	bh = max(5.0,priors[1]*np.exp(offset_tab[4]))
	return float(bx), float(by), float(bw), float(bh)



@jit(nopython=False, cache=True, fastmath=False)
def tile_filter(c_pred, c_box, c_tile, nb_box, prob_obj_cases, 
				patch, val_med_lim, val_med_obj, hist_count, yolo_nb_reg,
				nb_param, image_size):
	c_nb_box = 0
	for i in range(0,yolo_nb_reg):
		for j in range(0,yolo_nb_reg):
			kept_count = 0
			for k in range(0,nb_box):
				offset = int(k*(8+nb_param))
				c_box[4] = c_pred[offset+6,i,j] #probability
				c_box[5] = c_pred[offset+7,i,j] #objectness
				#Manual objectness penality on the edges of the images (help for both obj selection and NMS)
				if((j == 0 or j == yolo_nb_reg-1 or i == 0 or i == yolo_nb_reg-1)):
					c_box[4] = max(0.03,c_box[4]-0.05)
					c_box[5] = max(0.03,c_box[5]-0.05)
				
				if(c_box[5] >= prob_obj_cases[k]):
					bx = (c_pred[offset+0,i,j] + c_pred[offset+3,i,j])*0.5
					by = (c_pred[offset+1,i,j] + c_pred[offset+4,i,j])*0.5
					bw = max(5.0, c_pred[offset+3,i,j] - c_pred[offset+0,i,j])
					bh = max(5.0, c_pred[offset+4,i,j] - c_pred[offset+1,i,j])
					
					c_box[0] = bx - bw*0.5; c_box[1] = by - bh*0.5
					c_box[2] = bx + bw*0.5; c_box[3] = by + bh*0.5
					
					xmin = max(0,int(c_box[0]-5)); xmax = min(image_size,int(c_box[2]+5))
					ymin = max(0,int(c_box[1]-5)); ymax = min(image_size,int(c_box[3]+5))
					
					#Remove false detections over very large and very bright sources
					med_val_box = np.median(patch[ymin:ymax,xmin:xmax])
					if((med_val_box > val_med_lim[0] and c_box[5] < val_med_obj[0]) or\
					   (med_val_box > val_med_lim[1] and c_box[5] < val_med_obj[1]) or\
					   (med_val_box > val_med_lim[2] and c_box[5] < val_med_obj[2])):
						continue
					
					c_box[6] = k
					c_box[7:7+nb_param] = c_pred[offset+8:offset+8+nb_param,i,j]
					c_box[-1] = i*yolo_nb_reg+j
					c_tile[c_nb_box,:] = c_box[:]
					c_nb_box += 1
					kept_count += 1
					
			hist_count[kept_count] += 1
			
	return c_nb_box




@jit(nopython=True, cache=True, fastmath=False)
def first_NMS(c_tile, c_tile_kept, c_box, c_nb_box, box_prior_class,
				nms_thresholds, obj_thresholds):
	"""
	First NMS step: keep the box with the highest probability for each class

	INPUT:
	--------
	@c_tile: 2D array - Tile of detections
	@c_tile_kept: 2D array - Tile of detections kept after NMS
	@c_box: 1D array - Box
	@c_nb_box: int - Number of boxes
	@box_prior_class: 1D array - Box prior class
	@nms_thresholds: 1D array - NMS thresholds
	@obj_thresholds: 1D array - Objectness thresholds

	OUTPUT:
	--------
	@c_nb_box_final: int - Number of boxes after first NMS

	"""
	c_nb_box_final = 0
	is_match = 1
	c_box_size_prev = c_nb_box
	
	while(c_nb_box > 0):
		max_objct = np.argmax(c_tile[:c_box_size_prev,5])
		c_box = np.copy(c_tile[max_objct])
		c_tile[max_objct,5] = 0.0
		c_tile_kept[c_nb_box_final] = c_box
		c_nb_box_final += 1
		c_nb_box -= 1
		i = 0
		for i in range(0,c_box_size_prev):
			if(c_tile[i, 5] < 0.0000000001):
				continue
			IoU = fct_DIoU(c_box[:4], c_tile[i,:4])
			if(((IoU > nms_thresholds[0] and c_tile[i,5] < obj_thresholds[0]) or
			    (IoU > nms_thresholds[1] and c_tile[i,5] < obj_thresholds[1]) or
			    (IoU > nms_thresholds[2] and c_tile[i,5] < obj_thresholds[2]) or
			    (IoU > nms_thresholds[3] and c_tile[i,5] < obj_thresholds[3]))):
				c_tile[i, 5] = 0.0
				c_nb_box -= 1
				
	return c_nb_box_final


@jit(nopython=True, cache=True, fastmath=False)
def remove_extended(patch, c_tile, c_tile_kept, c_box, c_nb_box, nb_box,
						val_med_lim, val_med_obj):
	"""
	Remove detections that are too large or too bright compared to the median
	
	INPUT:
	--------
	@patch: 2D array - Full image patch
	@c_tile: 2D array - Tile of detections
	@c_tile_kept: 2D array - Tile of detections kept after NMS
	@c_box: 1D array - Box
	@c_nb_box: int - Number of boxes
	@nb_box: int - Number of boxes
	@val_med_lim: 1D array - Median limit for the patch
	@val_med_obj: 1D array - Median limit for the object

	OUTPUT:
	--------
	@c_nb_box_final: int - Number of boxes after removing extended detections

	"""

	c_nb_box_final = 0

	for i in range(0, c_nb_box):
		c_box = np.copy(c_tile[i])
	
		xmin = max(0,int(c_box[0]-5))
		ymin = max(0,int(c_box[1]-5))
		xmax = min(image_size,int(c_box[2]+5))
		ymax = min(image_size,int(c_box[3]+5))
		s_prior = c_box[6]
		
		#Remove small false detections over very large and very bright objects
		med_val_box = np.median(patch[ymin:ymax,xmin:xmax])
		
		if(((med_val_box > val_med_lim[0] and c_box[5] < val_med_obj[0]) or\
		   (med_val_box > val_med_lim[1] and c_box[5] < val_med_obj[1]) or\
		   (med_val_box > val_med_lim[2] and c_box[5] < val_med_obj[2]))):
			continue
		
		#Remove unlikely large detection over very large and very bright objects
		if(s_prior == nb_box-1 and (med_val_box > 0.2 and c_box[5] < 0.9)):
			continue
		
		c_tile_kept[c_nb_box_final] = c_box
		c_nb_box_final += 1
	return c_nb_box_final


@jit(nopython=True, cache=True, fastmath=False)
def second_NMS_local(boxes, comp_boxes, c_tile, direction, nms_threshold):
	c_tile[:,:] = 0.0
	nb_box_kept = 0

	mask_keep = np.where((boxes[:,0] > overlap) & (boxes[:,2] < patch_shift) &\
					(boxes[:,1] > overlap) & (boxes[:,3] < patch_shift))[0]
	
	mask_remain = np.where((boxes[:,0] <= overlap) | (boxes[:,2] >= patch_shift) |\
					(boxes[:,1] <= overlap) | (boxes[:,3] >= patch_shift))[0]
	
	nb_box_kept = np.shape(mask_keep)[0]
	c_tile[0:nb_box_kept,:] = boxes[mask_keep,:]
	
	
	comp_boxes[:,0:4] += direction*patch_shift
	
	comp_mask_keep = np.where((comp_boxes[:,0] < image_size) & (comp_boxes[:,2] > 0) &\
					(comp_boxes[:,1] < image_size) & (comp_boxes[:,3] > 0))[0]
	
	for b_ref in mask_remain:
		found = 0
		for b_comp in comp_mask_keep:
			IoU = fct_DIoU(boxes[b_ref,:4], comp_boxes[b_comp,:4])
			if(IoU > nms_threshold and boxes[b_ref,5] < comp_boxes[b_comp,5]):
				found = 1
				break
		if(found == 0):
			c_tile[nb_box_kept,:] = boxes[b_ref,:]
			nb_box_kept += 1
		
	return nb_box_kept
