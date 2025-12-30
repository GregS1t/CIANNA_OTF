
from utils.aux_fct import *

from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u


#Creating custom sample selection 
def dataset_perscut(dataset_path, out_file, out_format):

	full_cat = np.loadtxt(dataset_path, skiprows=18)

	#print ("Orig. Dataset size: ", np.shape(full_cat))

	c = SkyCoord(ra=full_cat[:,1]*u.degree, dec=full_cat[:,2]*u.degree, frame='icrs')
	x, y = utils.skycoord_to_pixel(c, wcs_img)
	xbeam, ybeam = utils.skycoord_to_pixel(c, wcs_beam)
	xbeam = xbeam.astype(int)
	ybeam = ybeam.astype(int)
	beamval = data_beam[xbeam,ybeam]

	#Get the "apparent flux" to correspond to the visible flux in the beam convolved input image
	flux_beam = full_cat[:,5]*beamval

	#The provided bmaj and bmin are the FWHM of a gaussian fitting of the sources
	#We use twice these dimensions to define our source sizes and clip the minimum size
	bmaj_pix = np.sqrt(full_cat[:,7]**2+beam_size**2)*to_sigma*4.0/(3600.0*pixel_size)
	bmin_pix = np.sqrt(full_cat[:,8]**2+beam_size**2)*to_sigma*4.0/(3600.0*pixel_size)
	
	surf = np.pi * bmaj_pix[:]*0.5 * bmin_pix[:]*0.5
	
	log_flux = np.log(flux_beam)
	log_bright = np.log(flux_beam/surf)
	
	index = np.where((
					((surf < 30.0) & (log_bright > -16.2)) |
					(((surf >= 30.0) & (surf < 400.0)) & ((log_bright > -16.0) | (log_flux > -12.7))) |
					((surf >= 300.0) & ((log_bright > -15.8) | (log_flux > -9.6)))
					))

	print ("Dataset size after selection function: ", np.shape(index)[1])
	
	new_cat = full_cat[index]
	
	if(out_format == 0):
		np.savetxt(out_file, new_cat,
			fmt="%d %.8f %.8f %.8f %.8f %.6g %.8f %.3f %.3f %.3f %d %d %d %.3f %.3f")
	else:
		np.savetxt(out_file, new_cat,
			fmt="%d %.8f %.8f %.8f %.8f %.6g %.8f %.3f %.3f %.3f %d %d")


def init_data_gen():

	global min_ra_train_pix, max_ra_train_pix, min_dec_train_pix
	global cut_data, coords, flux_list, bmaj_list, bmin_list, pa_list, diff_list
	global area_width, area_height, noise_area_width, noise_area_height
	global norm_data, norm_data_noise_1, norm_data_noise_2, norm_flux_data
	global input_data, targets, input_valid, targets_valid
	global full_cat_loaded, lims
	
	######################################################################
	#####                  TRAINING AREA DEFINITION                  #####
	######################################################################

	# From the scoring pipeline -> not available at the time of the challenge
	#560: {"ra_min": -0.6723, "ra_max": 0.0, "dec_min": -29.9400, "dec_max": -29.4061},

	#Loading the result of the training selection function
	train_list = np.loadtxt("TrainingSet_perscut.txt")

	#Define the training zone in RA, DEC from the training source catalog
	min_ra_train = np.min(train_list[:,1]); max_ra_train = np.max(train_list[:,1])
	min_dec_train = np.min(train_list[:,2]); max_dec_train = np.max(train_list[:,2])

	#Define the training part of the image as the area containing the training examples
	c_min = SkyCoord(ra=min_ra_train*u.degree, dec=min_dec_train*u.degree, frame='icrs')
	c_max = SkyCoord(ra=max_ra_train*u.degree, dec=max_dec_train*u.degree, frame='icrs')

	#Convert the training zone coordinates in image pixel coordinate
	min_ra_train_pix, min_dec_train_pix = utils.skycoord_to_pixel(c_min, wcs_img)
	max_ra_train_pix, max_dec_train_pix = utils.skycoord_to_pixel(c_max, wcs_img)
	
	#Prevent some non-labeled edge sources to appear in training examples
	min_ra_train_pix -= 10; max_ra_train_pix += 10
	min_dec_train_pix += 10; max_dec_train_pix -= 10
	
	print ("\nTraining area edges:")
	print (min_ra_train, max_ra_train, min_dec_train, max_dec_train)
	print (min_ra_train_pix, max_ra_train_pix, min_dec_train_pix, max_dec_train_pix)
	
	area_width = (min_ra_train_pix - max_ra_train_pix)
	area_height = (max_dec_train_pix - min_dec_train_pix)
	print ("Training area size (pixels)")
	print (area_width, area_height)
	
	
	######################################################################
	#####                 AUX EMPTY AREA DEFINITION                  #####
	######################################################################
	
	#Define two zones of the whole unlabeled-image where there should be no detectable source anymore
	min_ra_noise = [-2.8,2.2]; max_ra_noise = [-2.2,2.8]
	min_dec_noise = [-28.7,-32.3]; max_dec_noise = [-28.1,-31.7]
	
	#Convert the two zones coordinates in image pixel coordinate
	c_min_noise_1 = SkyCoord(ra=min_ra_noise[0]*u.degree, dec=min_dec_noise[0]*u.degree, frame='icrs')
	c_max_noise_1 = SkyCoord(ra=max_ra_noise[0]*u.degree, dec=max_dec_noise[0]*u.degree, frame='icrs')
	
	c_min_noise_2 = SkyCoord(ra=min_ra_noise[1]*u.degree, dec=min_dec_noise[1]*u.degree, frame='icrs')
	c_max_noise_2 = SkyCoord(ra=max_ra_noise[1]*u.degree, dec=max_dec_noise[1]*u.degree, frame='icrs')
	
	min_ra_noise_pix_1, min_dec_noise_pix_1 = utils.skycoord_to_pixel(c_min_noise_1, wcs_img)
	max_ra_noise_pix_1, max_dec_noise_pix_1 = utils.skycoord_to_pixel(c_max_noise_1, wcs_img)
	
	min_ra_noise_pix_2, min_dec_noise_pix_2 = utils.skycoord_to_pixel(c_min_noise_2, wcs_img)
	max_ra_noise_pix_2, max_dec_noise_pix_2 = utils.skycoord_to_pixel(c_max_noise_2, wcs_img)

	print ("\nEmpty zone 1 edges:")
	print(min_ra_noise_pix_1, min_dec_noise_pix_1)
	print(max_ra_noise_pix_1, max_dec_noise_pix_1)

	print ("\nEmpty zone 2 edges:")
	print(min_ra_noise_pix_2, min_dec_noise_pix_2)
	print(max_ra_noise_pix_2, max_dec_noise_pix_2)

	#identical for both noise regions
	noise_area_width = (min_ra_noise_pix_1 - max_ra_noise_pix_1)
	noise_area_height = (max_dec_noise_pix_1 - min_dec_noise_pix_1)

	#Get 3 cutouts from the full image: The training cutout, and two "no-sources / noise-only" cutouts
	cut_data = full_img[int(min_dec_train_pix):int(max_dec_train_pix),\
						 int(max_ra_train_pix):int(min_ra_train_pix)]

	cut_data_noise_1 = full_img[int(min_dec_noise_pix_1):int(max_dec_noise_pix_1),\
						 int(max_ra_noise_pix_1):int(min_ra_noise_pix_1)]
						 
	cut_data_noise_2 = full_img[int(min_dec_noise_pix_2):int(max_dec_noise_pix_2),\
						 int(max_ra_noise_pix_2):int(min_ra_noise_pix_2)]


	######################################################################
	#####             TRAINING SOURCE CATALOG DEFINITION             #####
	######################################################################	
	
	#Get the sky coordinate of all sources in the selected training catalog
	c = SkyCoord(ra=train_list[:,1]*u.degree, dec=train_list[:,2]*u.degree, frame='icrs')
	x, y = utils.skycoord_to_pixel(c, wcs_img)

	#Compute the bmaj and bmin in pixel size (approximate) -> Only used to define the bouding boxes	
	#bmaj = np.sqrt(train_list[:,7]**2+beam_size**2)*to_sigma*4.0/(3600.0*pixel_size)
	#bmin = np.sqrt(train_list[:,8]**2+beam_size**2)*to_sigma*4.0/(3600.0*pixel_size)

	n_w       = np.zeros((np.shape(train_list)[0]))
	n_h       = np.zeros((np.shape(train_list)[0]))
	coords    = np.zeros((np.shape(train_list)[0],4))
	flux_list = np.zeros((np.shape(train_list)[0]))
	bmaj_list = np.zeros((np.shape(train_list)[0]))
	bmin_list = np.zeros((np.shape(train_list)[0]))
	pa_list   = np.zeros((np.shape(train_list)[0]))
	diff_list = np.zeros((np.shape(train_list)[0]))

	flux_list[:] = train_list[:,5]
	bmaj_list[:] = train_list[:,7]
	bmin_list[:] = train_list[:,8]
	pa_list[:]   = train_list[:,9]
	
	index0 = np.where(bmaj_list[:] < beam_size)
	index1 = np.where(bmin_list[:] < beam_size)
	
	med_unr_bmaj = np.median(bmaj_list[index0])
	med_unr_bmin = np.median(bmin_list[index1])
	
	#Remap all the PA values so they are all in the range [-90,90]
	index = np.where((pa_list[:] > 90.0) & (pa_list[:] <= 270.0))
	pa_list[index[0]] = -90.0 + (pa_list[index[0]] - 90.0)
	index = np.where((pa_list[:] > 270.0) & (pa_list[:] < 360.0))
	pa_list[index[0]] = -90.0 + (pa_list[index[0]] - 270.0)

	w = train_list[:,7]/(3600.0*pixel_size)*2
	h = train_list[:,8]/(3600.0*pixel_size)*2

	#Convert all bmaj, bmin size onto regular "boxes" as defined for classical detection task
	#construct the smallest box that contain the ellipse
	#ellipses_to_boxes(np.shape(train_list)[0], w, h, pa_list, n_w, n_h)
	
	
	for i in range(0,np.shape(train_list)[0]):
		W = w[i]
		H = h[i]
		vertices = np.array([[-W*0.5,-H*0.5],[-W*0.5,H*0.5],[W*0.5, -H*0.5],[W*0.5,H*0.5]])

		vertices_new = np.zeros((4,2))
		vertices_new[:,0] = np.cos(train_list[i,9]*np.pi/180.0)*vertices[:,0] + np.sin(train_list[i,9]*np.pi/180.0)*vertices[:,1]
		vertices_new[:,1] = - np.sin(train_list[i,9]*np.pi/180.0)*vertices[:,0] + np.cos(train_list[i,9]*np.pi/180.0)*vertices[:,1]

		n_w[i] = max(vertices_new[:,0]) - min(vertices_new[:,0])
		n_h[i] = max(vertices_new[:,1]) - min(vertices_new[:,1])
	

	#n_w = np.copy(w)
	#n_h = np.copy(w)

	#Clip the too small boxes (in pixel size)
	n_w = np.clip(n_w, 5.0, 64.0)
	n_h = np.clip(n_h, 5.0, 64.0)
	
	#plt.hist(np.log(n_w), bins=100)
	#plt.yscale("log")
	#plt.show()

	#Convert the positions and sizes into coordinates inside the full image
	coords[:,0] = x - n_w[:]*0.5 + 0.5
	coords[:,1] = x + n_w[:]*0.5 + 0.5
	coords[:,2] = (map_pixel_size - (y + 1.0)) - n_h[:]*0.5
	coords[:,3] = (map_pixel_size - (y + 1.0)) + n_h[:]*0.5
	
	#Get the "apparent flux" to correspond to the visible flux in the beam convolved map
	#Require to get the value of the beam for each source position (approximation)
	xbeam, ybeam = utils.skycoord_to_pixel(c, wcs_beam)
	xbeam = xbeam.astype(int); ybeam = ybeam.astype(int)
	beamval=data_beam[xbeam,ybeam]
	
	print ("Log10 Flux:", np.log10(np.min(flux_list)), np.log10(np.max(0.05*flux_list)))
	flux_list[:] = flux_list[:]*beamval[:]
	
	#Cap the minimum and maximum value for the Flux, Bmaj and Bmin for the regression 
	#(not linked to the bounding boxes)
	flux_list = np.clip(flux_list, np.min(flux_list),0.05*np.max(flux_list))
	bmaj_list = np.clip(bmaj_list, 0.9, 60.0) # In arcsec
	bmin_list = np.clip(bmin_list, 0.3, 30.0) # In arcsec
	
	print ("Flux clipping: ", np.min(flux_list), np.max(flux_list))
	print ("Bmaj clipping: ", np.min(bmaj_list), np.max(bmaj_list))
	print ("Bmin clipping: ", np.min(bmin_list), np.max(bmin_list))
	
	#Flag very small objects for which PA estimation is too difficult and set their target PA to 0
	small_id = np.where(bmaj_list[:] <= 1.8) #In arcsec = 3.0 pixels
	pa_list[small_id] = 0.0

	#Switch to logscale for Flux, Bmaj, Bmin to obtain flatter distributions across scales
	flux_list = np.log(flux_list)
	bmaj_list = np.log(bmaj_list)
	bmin_list = np.log(bmin_list)

	#Get the normalization limits and save them to convert network predictions back to physical quantities
	flux_min = np.min(flux_list); flux_max = np.max(flux_list)
	bmaj_min = np.min(bmaj_list); bmaj_max = np.max(bmaj_list)
	bmin_min = np.min(bmin_list); bmin_max = np.max(bmin_list)

	lims = np.zeros((3,2))
	lims[0] = [flux_max, flux_min]
	lims[1] = [bmaj_max, bmaj_min]
	lims[2] = [bmin_max, bmin_min]

	print ("\nMin-Max values used for normalization (Flux, Bmaj, Bmin):")
	print (lims[0], lims[1], lims[2])
	np.savetxt("train_cat_lims.txt", lims)
	
	#Normalize the extra parameters
	flux_list[:] = (flux_list[:] - flux_min)/(flux_max - flux_min)
	bmaj_list[:] = (bmaj_list[:] - bmaj_min)/(bmaj_max - bmaj_min)
	bmin_list[:] = (bmin_list[:] - bmin_min)/(bmin_max - bmin_min)
	
	print ("\nNew mean, std and Min/Max of the distribution for each quantity:")
	print(np.mean(flux_list), np.std(flux_list), np.min(flux_list), np.max(flux_list))
	print(np.mean(bmaj_list), np.std(bmaj_list), np.min(bmaj_list), np.max(bmaj_list))
	print(np.mean(bmin_list), np.std(bmin_list), np.min(bmin_list), np.max(bmin_list))
	
	
	######################################################################
	#####             Network input data normalization               #####
	######################################################################	
	
	print("\nRaw cube Min/Max values:", np.min(cut_data), np.max(cut_data))
	
	#Normalize all possible input fields using a tanh scaling
	cut_data = np.clip(cut_data,min_pix,max_pix)
	norm_data = (cut_data - min_pix) / (max_pix-min_pix)
	norm_data = np.tanh(3.0*norm_data)
	
	cut_data_noise_1 = np.clip(cut_data_noise_1,min_pix,max_pix)
	norm_data_noise_1 = (cut_data_noise_1 - min_pix) / (max_pix-min_pix)
	norm_data_noise_1 = np.tanh(3.0*norm_data_noise_1)
	
	cut_data_noise_2 = np.clip(cut_data_noise_2,min_pix,max_pix)
	norm_data_noise_2 = (cut_data_noise_2 - min_pix) / (max_pix-min_pix)
	norm_data_noise_2 = np.tanh(3.0*norm_data_noise_2)
	
	input_data = np.zeros((nb_images,image_size*image_size*im_depth), dtype="float32")
	targets = np.zeros((nb_images,1+max_nb_obj_per_image*(7+nb_param+1)), dtype="float32")
	
	input_valid = np.zeros((nb_valid,image_size*image_size*im_depth), dtype="float32")
	targets_valid = np.zeros((nb_valid,1+max_nb_obj_per_image*(7+nb_param+1)), dtype="float32")


## Data augmentation
def create_train_batch(visual=0):
	#Construct a randomly augmented batch from the training area and source catalog
	
	for i in range(0, nb_images):

		#####      RANDOM POSITION IN TRAINING REGION      #####
		if(np.random.rand() > add_noise_prop):
		
			#Select a random position inside the traning area
			p_y = np.random.randint(0,area_width-image_size)
			p_x = np.random.randint(0,area_height-image_size)
			
			if(visual):
				print(p_y,p_x)
			
			#The patch is verticaly flipped so the origin of the image is top left
			patch = np.flip(np.copy(norm_data[p_x:p_x+image_size,\
									 p_y:p_y+image_size]),axis=0)
			
			#Randomly set the image to be flipped (hor/vert) or rotated (-90,+90)
			flip_w = 0; flip_h = 0
			
			rot_90 = 0
			rot_rand = 1.0 #np.random.random()
			if(rotate_flag and rot_rand < 0.33):
				rot_90 = -1
				patch = np.rot90(patch, k=-1, axes=(0,1))
			elif(rotate_flag and rot_rand < 0.66):
				rot_90 = 1
				patch = np.rot90(patch, k=1, axes=(0,1))
			
			if(np.random.random() < flip_hor):
				flip_w = 1
				patch = np.flip(patch, axis=1)
			if(np.random.random() < flip_vert):
				flip_h = 1
				patch = np.flip(patch, axis=0)
			
			#The input is flatten to be in the proper format for CIANNA
			input_data[i,:] = patch.flatten("C")
			
			#Find all boxes that overlap the selected input image
			patch_boxes_id = np.where((coords[:,1] > max_ra_train_pix + p_y) &\
					(coords[:,0] < max_ra_train_pix + p_y + image_size) &\
					(map_pixel_size - coords[:,2] > min_dec_train_pix + p_x) &\
					(map_pixel_size - coords[:,3] < min_dec_train_pix + p_x + image_size))[0]
			
			
			orig_l_coords = coords[patch_boxes_id]
			box_flux = flux_list[patch_boxes_id]
			
			#Convert to local image coordinate
			orig_l_coords[:,0:2] -= (max_ra_train_pix + p_y)
			orig_l_coords[:,2:4] = image_size + ((orig_l_coords[:,2:4] - map_pixel_size) + (min_dec_train_pix + p_x))
			cut_l_coords = np.clip(orig_l_coords, 0, image_size)
			
			#Remove or flag as difficult objects that are too close to the edges			
			frac_in = (abs(cut_l_coords[:,0]-cut_l_coords[:,1])*abs(cut_l_coords[:,2]-cut_l_coords[:,3])) \
				/ (abs(orig_l_coords[:,0]-orig_l_coords[:,1])*abs(orig_l_coords[:,2]-orig_l_coords[:,3]))
			
			c_w = abs(cut_l_coords[:,0]-cut_l_coords[:,1])
			c_h = abs(cut_l_coords[:,2]-cut_l_coords[:,3])
			
			in_box_id = np.where((c_w >= 4.0) & (c_h >= 4.0))[0]
			
			patch_boxes_id = patch_boxes_id[in_box_id]
			
			orig_l_coords = orig_l_coords[in_box_id]
			cut_l_coords = cut_l_coords[in_box_id]
			frac_in = (abs(cut_l_coords[:,0]-cut_l_coords[:,1])*abs(cut_l_coords[:,2]-cut_l_coords[:,3])) \
				/ (abs(orig_l_coords[:,0]-orig_l_coords[:,1])*abs(orig_l_coords[:,2]-orig_l_coords[:,3]))
			
			diff_box_id_2 = np.where((frac_in >= 0.5) & (frac_in <= 1.0))[0]
			diff_box_id_3 = np.where((frac_in < 0.5))[0]
			
			pa_kept = np.copy(pa_list[patch_boxes_id[:]])
			diff_kept = np.copy(diff_list[patch_boxes_id[:]])
			
			diff_kept[diff_box_id_2] = 0
			diff_kept[diff_box_id_3] = 1
			
			#Final box selection
			keep_box_coords = np.copy(coords[patch_boxes_id])
			
			#Convert to local image coordinate
			keep_box_coords[:,0:2] -= (max_ra_train_pix + p_y)
			keep_box_coords[:,2:4] = image_size + ((keep_box_coords[:,2:4] - map_pixel_size) + (min_dec_train_pix + p_x))
			
			if(rot_90 == -1):
				mod_keep_box_coords_x = (image_size) - np.copy(keep_box_coords[:,2:4])
				mod_keep_box_coords_y = np.copy(keep_box_coords[:,0:2])
				keep_box_coords[:,0:2] = mod_keep_box_coords_x
				keep_box_coords[:,2:4] = mod_keep_box_coords_y
			elif(rot_90 == 1):
				mod_keep_box_coords_x = np.copy(keep_box_coords[:,2:4])
				mod_keep_box_coords_y = (image_size) - np.copy(keep_box_coords[:,0:2])
				keep_box_coords[:,0:2] = mod_keep_box_coords_x
				keep_box_coords[:,2:4] = mod_keep_box_coords_y
			
			keep_box_coords[:,0:2] = flip_w*(image_size) + np.sign(0.5-flip_w)*keep_box_coords[:,0:2]
			keep_box_coords[:,2:4] = flip_h*(image_size) + np.sign(0.5-flip_h)*keep_box_coords[:,2:4]
			
			#Add random uncertainty in target position as regularization to prevent over-training
			pos_offset = np.random.normal(0.0,0.15,(np.shape(keep_box_coords)[0],2))
			flux_offset_rel = np.random.normal(1.0,0.1,(np.shape(keep_box_coords)[0]))
			flux_offset_abs = np.random.normal(0.0,0.05,(np.shape(keep_box_coords)[0]))
			flux_offset_rel[:] = 1.0
			flux_offset_abs[:] = 0.0

			keep_box_coords[:,0] += pos_offset[:,0]; keep_box_coords[:,1] += pos_offset[:,0]
			keep_box_coords[:,2] += pos_offset[:,1]; keep_box_coords[:,3] += pos_offset[:,1]

			keep_box_coords = np.clip(keep_box_coords, 0, image_size)
			
			if(rot_90 == -1):
				pa_kept[:] = -np.sign(pa_kept)*(90.0-np.abs(pa_kept[:]))
			elif(rot_90 == 1):
				pa_kept[:] = -np.sign(pa_kept)*(90.0-np.abs(pa_kept[:]))
			
			if(flip_h):
				pa_kept[:] = -pa_kept[:]
			if(flip_w):
				pa_kept[:] = -pa_kept[:]
			
			targets[i,:] = 0.0
			targets[i,0] = min(max_nb_obj_per_image, np.shape(patch_boxes_id)[0])
			for k in range(0,int(targets[i,0])):
				
				xmin = min(keep_box_coords[k,0:2])
				xmax = max(keep_box_coords[k,0:2])
				ymin = min(keep_box_coords[k,2:4])
				ymax = max(keep_box_coords[k,2:4])
				
				targets[i,1+k*(7+nb_param+1):1+(k+1)*(7+nb_param+1)] = \
						np.array([1.0,xmin,ymin,0.0,xmax,ymax,1.0,\
						flux_list[int(patch_boxes_id[k])]*flux_offset_rel[k]+flux_offset_abs[k],\
						bmaj_list[int(patch_boxes_id[k])],\
						bmin_list[int(patch_boxes_id[k])],\
						np.cos(pa_kept[k]*np.pi/180.0),\
						(np.sin(pa_kept[k]*np.pi/180.0)+1.0)*0.5,\
						diff_kept[k]])
				
		else:
			#####      RANDOM POSITION IN NOISE REGIONS      #####
			p_y = np.random.randint(0,3000-image_size)
			p_x = np.random.randint(0,3500-image_size)
			
			#No target sources in this region (which is a simplification)
			keep_box_coords = np.empty(0)

			#Select one of the two noise region
			if(np.random.rand() > 0.5):
				patch = np.flip(np.copy(norm_data_noise_1[p_x:p_x+image_size,\
									 p_y:p_y+image_size]),axis=0)
			else:
				patch = np.flip(np.copy(norm_data_noise_2[p_x:p_x+image_size,\
									 p_y:p_y+image_size]),axis=0)
			
			#Randomly set the image to be flipped (hor/vert) or rotated (-90,+90)
			flip_w = 0; flip_h = 0
			
			rot_90 = 0
			rot_rand = 1.0 #np.random.random()
			if(rotate_flag and rot_rand < 0.33):
				rot_90 = -1
				patch = np.rot90(patch, k=-1, axes=(0,1))
			elif(rotate_flag and rot_rand < 0.66):
				rot_90 = 1
				patch = np.rot90(patch, k=1, axes=(0,1))
			
			if(np.random.random() < flip_hor):
				flip_w = 1
				patch = np.flip(patch, axis=1)
			if(np.random.random() < flip_vert):
				flip_h = 1
				patch = np.flip(patch, axis=0)
			
			input_data[i,:] = patch.flatten("C")
			
			targets[i,:] = 0.0
		
		#Visualisation of the generated images
		if(visual and np.shape(keep_box_coords)[0] > 0):
			fig, ax = plt.subplots(1,1, figsize=(8,8), dpi=200, constrained_layout=True)
			ax.imshow(patch, cmap="hot", vmin=0.0, vmax=0.2*1.0, extent=(0,image_size,image_size,0))
			plt.scatter((keep_box_coords[:,1]+keep_box_coords[:,0])*0.5, (keep_box_coords[:,3]+keep_box_coords[:,2])*0.5, s=0.5, marker="+")
			for k in range(0, np.shape(keep_box_coords)[0]):
				if(diff_kept[k] > 0):
					el = patches.Rectangle((keep_box_coords[k,0],keep_box_coords[k,2]), 
						(keep_box_coords[k,1]-keep_box_coords[k,0]), (keep_box_coords[k,3]-keep_box_coords[k,2]), 
						linewidth=0.3, fill=False, color="lightgreen", zorder=3, ls="--")
					ax.add_patch(el)
				else:
					el = patches.Rectangle((keep_box_coords[k,0],keep_box_coords[k,2]), 
						(keep_box_coords[k,1]-keep_box_coords[k,0]), (keep_box_coords[k,3]-keep_box_coords[k,2]), 
						linewidth=0.3, fill=False, color="lightgreen", zorder=3)
					ax.add_patch(el)	
					#el = patches.Ellipse(((keep_box_coords[k,0]+keep_box_coords[k,1])*0.5,(keep_box_coords[k,2]+keep_box_coords[k,3])*0.5), 
					#	np.sqrt((np.exp(bmaj_list[int(patch_boxes_id[k])]*(lims[1,0]-lims[1,1])+lims[1,1]))**2+beam_size**2)*to_sigma*3.0/(3600.0*pixel_size),
					#	np.sqrt((np.exp(bmin_list[int(patch_boxes_id[k])]*(lims[2,0]-lims[2,1])+lims[2,1]))**2+beam_size**2)*to_sigma*3.0/(3600.0*pixel_size),
					#	pa_kept[k],
					#	linewidth=0.3, fill=False, color="blue", zorder=3, ls="-")
					#ax.add_patch(el)
			
			plt.show()
		
	return input_data, targets


def create_valid_batch(visual=0):
	# Define a "static" regularly sampled "valid/test" dataset
	# Note: This dataset is not "distinct" from the training dataset in the sense that it is defined on the same training area.
	# This is not sufficient to properly monitor overtraining, but it is acceptable in the present context due to the presence
	# of the "scorer" on the full image (minus the training area) that is used afterward as a real "valid/test" dataset.

	patch_shift = 256

	for i in range(0, nb_valid):
		
		p_y = int(i/10); p_x = int(i%10)
		
		#The patch is verticaly flipped so the origin of the image is top left
		patch = np.flip(np.copy(norm_data[p_x*patch_shift:p_x*patch_shift+image_size,\
								 p_y*patch_shift:p_y*patch_shift+image_size]),axis=0)
		
		#The input is flatten to be in the proper format for CIANNA
		input_valid[i,:] = patch.flatten("C")
		
		#Find all boxes that overlap the selected input image
		patch_boxes_id = np.where((coords[:,1] > max_ra_train_pix + p_y*patch_shift) &\
					(coords[:,0] < max_ra_train_pix + p_y*patch_shift + image_size) &\
					(map_pixel_size - coords[:,2] > min_dec_train_pix + p_x*patch_shift) &\
					(map_pixel_size - coords[:,3] < min_dec_train_pix + p_x*patch_shift + image_size))[0]
		
		orig_l_coords = coords[patch_boxes_id]
		
		#Convert to local image coordinate
		orig_l_coords[:,0:2] -= (max_ra_train_pix + p_y*patch_shift)
		orig_l_coords[:,2:4] = image_size + ((orig_l_coords[:,2:4] - map_pixel_size) + (min_dec_train_pix + p_x*patch_shift))
		cut_l_coords = np.clip(orig_l_coords, 0, image_size)
		
		#Remove or flag as difficult objects that are too close to the edges
		
		frac_in = (abs(cut_l_coords[:,0]-cut_l_coords[:,1])*abs(cut_l_coords[:,2]-cut_l_coords[:,3])) \
			/ (abs(orig_l_coords[:,0]-orig_l_coords[:,1])*abs(orig_l_coords[:,2]-orig_l_coords[:,3]))
		
		c_w = abs(cut_l_coords[:,0]-cut_l_coords[:,1])
		c_h = abs(cut_l_coords[:,2]-cut_l_coords[:,3])
		
		in_box_id = np.where((c_w >= 4.0) & (c_h >= 4.0))[0]
		
		patch_boxes_id = patch_boxes_id[in_box_id]
		
		orig_l_coords = orig_l_coords[in_box_id]
		cut_l_coords = cut_l_coords[in_box_id]
		
		frac_in = (abs(cut_l_coords[:,0]-cut_l_coords[:,1])*abs(cut_l_coords[:,2]-cut_l_coords[:,3])) \
			/ (abs(orig_l_coords[:,0]-orig_l_coords[:,1])*abs(orig_l_coords[:,2]-orig_l_coords[:,3]))
		
		diff_box_id_2 = np.where((frac_in >= 0.5) & (frac_in <= 1.0))[0]
		diff_box_id_3 = np.where((frac_in < 0.5))[0]
		
		pa_kept = np.copy(pa_list[patch_boxes_id[:]])
		diff_kept = np.copy(diff_list[patch_boxes_id[:]])
		
		diff_kept[diff_box_id_2] = 0
		diff_kept[diff_box_id_3] = 1
		
		#Final box selection
		keep_box_coords = coords[patch_boxes_id]
		#Convert to local image coordinate
		keep_box_coords[:,0:2] -= (max_ra_train_pix + p_y*patch_shift)
		keep_box_coords[:,2:4] = image_size + ((keep_box_coords[:,2:4] - map_pixel_size) + (min_dec_train_pix + p_x*patch_shift))
		keep_box_coords = np.clip(keep_box_coords, 0, image_size)
		
		pa_kept = np.copy(pa_list[patch_boxes_id[:]])
		
		targets_valid[i,:] = 0.0
		targets_valid[i,0] = min(max_nb_obj_per_image, np.shape(patch_boxes_id)[0])
		for k in range(0,int(targets_valid[i,0])):
				
			xmin = min(keep_box_coords[k,0:2])
			xmax = max(keep_box_coords[k,0:2])
			ymin = min(keep_box_coords[k,2:4])
			ymax = max(keep_box_coords[k,2:4])
			
			targets_valid[i,1+k*(7+nb_param+1):1+(k+1)*(7+nb_param+1)] = \
					np.array([1.0,xmin,ymin,0.0,xmax,ymax,1.0,
					flux_list[patch_boxes_id[k]], bmaj_list[patch_boxes_id[k]],\
					bmin_list[patch_boxes_id[k]], np.cos(pa_kept[k]*np.pi/180.0),\
					(np.sin(pa_kept[k]*np.pi/180.0)+1.0)*0.5,\
					diff_kept[k]])
			
		if(visual):
			fig, ax = plt.subplots(1,1, figsize=(8,8), dpi=200, constrained_layout=True)
			ax.imshow(patch, cmap="hot", vmin=0.0, vmax=0.2*1.0)
			for k in range(0, np.shape(keep_box_coords)[0]):
				el = patches.Rectangle((keep_box_coords[k,0],keep_box_coords[k,2]), 
					(keep_box_coords[k,1]-keep_box_coords[k,0]), (keep_box_coords[k,3]-keep_box_coords[k,2]), 
					linewidth=0.3, fill=False, color="lightgreen", zorder=3)
				ax.add_patch(el)	
			
			plt.show()
	
	return input_valid, targets_valid

	

def create_full_pred():
	#Decompose the full SDC1 image into patches of the appropriate input size with partial overlap between them
	
	pred_all = np.zeros((nb_images_all,image_size*image_size*im_depth), dtype="float32")
	patch = np.zeros((image_size,image_size), dtype="float32")
	 
	# Get the full image and normalize it
	full_data_norm = np.clip(full_img,min_pix,max_pix)
	full_data_norm = (full_data_norm - min_pix) / (max_pix-min_pix)
	full_data_norm = np.tanh(3.0*full_data_norm)
	
	for i_d in range(0,nb_images_all):
	
		p_y = int(i_d/nb_area_w)
		p_x = int(i_d%nb_area_w)

		# Coordinates of the patch in the full image
		xmin = p_x*patch_shift - orig_offset
		xmax = p_x*patch_shift + image_size - orig_offset
		ymin = p_y*patch_shift - orig_offset
		ymax = p_y*patch_shift + image_size - orig_offset

		px_min = 0; px_max = image_size
		py_min = 0; py_max = image_size

		set_zero = 0

		# Check if the patch is out of the image
		if(xmin < 0):
			px_min = -xmin
			xmin = 0
			set_zero = 1
		if(ymin < 0):
			py_min = -ymin
			ymin = 0
			set_zero = 1
		if(xmax > map_pixel_size):
			px_max = image_size - (xmax-map_pixel_size)
			xmax = map_pixel_size
			set_zero = 1
		if(ymax > map_pixel_size):
			py_max = image_size - (ymax-map_pixel_size)
			ymax = map_pixel_size
			set_zero = 1

		if(set_zero):
			patch[:,:] = 0.0
		
		patch[px_min:px_max,py_min:py_max] = np.flip(full_data_norm[xmin:xmax,ymin:ymax],axis=0)
			
		pred_all[i_d,:] = patch.flatten("C")
	
	return pred_all

		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
	


