import sys, os, csv, datetime, math
import numpy as np
from scipy.stats import spearmanr
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
from tensorflow.python.client import device_lib
from tensorflow.contrib.tensorboard.plugins import projector
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, roc_curve, auc, precision_recall_curve, explained_variance_score, mean_absolute_error
from sklearn.svm import SVC, SVR

# from tensorflow.python import debug as tf_debug
from matplotlib import pyplot as plt

# bsub -R 'rusage[mem=185000,ngpus_excl_p=1]' -R 'select[gpu_model1==GeForceGTX1080Ti]' -o log.txt python new_cluster.py
for i in range(6): print()
print(device_lib.list_local_devices())
for i in range(6): print()
sys.stdout.flush()
current_dir = os.path.abspath(os.path.join(os.pardir, os.pardir, os.pardir))
# current_dir = os.getcwd()
# Target_dir = os.getcwd() + '/output_' + str(datetime.datetime.now().month) + '.' + str(datetime.datetime.now().day) + '.' + str(datetime.datetime.now().hour) + '/'
Target_dir = os.getcwd() + '/output/'
os.mkdir(Target_dir)
# metadata = os.path.join(Target_dir, 'metadata.tsv')




#################################################################################################
#################################################################################################
#################################################################################################
################################## settings and hyperparameters #################################
#################################################################################################
#################################################################################################
#################################################################################################





outprint, prog_bar, figures, do_sklearn = 1, 0, 0, 1
use_cols = [3,4,5,6,7,8,9,10,11,12,13,14,15,17,18]
          #[0,1,2,3,4,5,6, 7, 8, 9,10,11,12,13]

corr_col_names = ['SEX', 'AGE', 'BMI', 'fat_imp', 'insulin', 'HBA1C', 'diabetes', 'DI']
corr_cols = [3,4,7,8,9,13,15,17]

# 0 ....... subject_idout
# 1 ....... MRI_sessionid
# 2 ....... files_n
# 3 ....... SEX
# 4 ....... AGE
# 5 ....... height
# 6 ....... weight
# 7 ....... BMI
# 8 ....... fat_percentage_bioimpedance
# 9 ....... insulin_sensitivity
# 10 ...... total_adipose_tissue_liter
# 11 ...... visceral_adipose_tissue_liter
# 12 ...... adipose_tissue_upper_limbs_liter
# 13 ...... HBA1C
# 14 ...... diabetes_allmethods
# 15 ...... diabetes_HbA1c
# 16 ...... cluster
# 17 ...... DI


use_pre_diabetes = True

test_ratio = .3
train_data_split = 10
second_dense_layer, dense_units = 1, 64

nn_type = 'dense_net' #conv_net, dense_net

prediction_label = 'all' # diab_cont, di_index, insulin, diab_all
additional_features = False

batch_size = 6
total_epochs = 300


num_conv_layers = 4

# Network Parameters
growth_k = 6

first_maxpool_stride = 2
first_conv_filters = 4

num_dense_blocks = 3
num_layers_1st_dense = 4
num_layers_2nd_dense = 4 * num_layers_1st_dense
num_layers_3rd_dense = 2 * num_layers_2nd_dense
num_layers_4th_dense = 2 * num_layers_3rd_dense


#AdamOptimizer
init_learning_rate = 2e-4
lr_decay = .994
lr_amp = .33
use_cycles = True
adapt_lr_freq = 8
adapt_lr_ratio = .9
epsilon = 1e-8
dropout_rate = 0.3

#MomentumOptimizer
nesterov_momentum, weight_decay = 0.9, 1e-4


kernel_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False, dtype=tf.float32)
# kernel_init = tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32)
kernel_init_lin = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False, dtype=tf.float32)
# kernel_init_lin = tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32)


use_small_scans = True
if use_small_scans == True:
	directories = [current_dir + '/harddrive1', current_dir + '/harddrive2', current_dir + '/scans_small']
if use_small_scans == False:
	directories = [current_dir + '/harddrive1', current_dir + '/harddrive2', current_dir + '/scans_shaped']





#################################################################################################
#################################################################################################
#################################################################################################
###################################### function definitions #####################################
#################################################################################################
#################################################################################################
#################################################################################################






def printer(var, comment):
	print()
	print(comment)
	print(var)
	print()

def load_body_array(Session_ID, shrinker=3):

	body = np.asarray(np.load(directories[2] + '/scan_' + str(Session_ID) + '.npy'))

	if use_small_scans == False:

		height_dim = 95
		rows, cols = body.shape[1:]

		if body.shape[0] != height_dim:

			dummy = np.zeros([height_dim, rows, cols])
			pre_vec = np.arange(body.shape[0])
			post_vec = np.linspace(0, body.shape[0]-1, height_dim)

			for row in range(body.shape[1]):

				for col in range(body.shape[2]):

					f = interp1d(pre_vec, body[:,row,col], kind='nearest')

					dummy[:,row,col] = f(post_vec)

			body = dummy


	body_hist_vals, body_bin_edges = np.histogram(np.reshape(body, [-1]), 100)


	body[body <= body_bin_edges[18]] = 0.


	BodyMean = np.mean(np.reshape(body[body > 0.], [-1]))

	body[body > 0.] -= BodyMean

	BodySTD = np.std(np.reshape(body[body > 0.], [-1]))

	body /= BodySTD


	body_hist_vals2, body_bin_edges2 = np.histogram(np.reshape(body, [-1]), 100)

	cutoff = np.reshape(np.asarray(np.where(body_hist_vals2 >= 500)), [-1])

	body[body < body_bin_edges2[cutoff[0]]] = body_bin_edges2[cutoff[0]]
	body[body > body_bin_edges2[cutoff[-1]]] = body_bin_edges2[cutoff[-1]]




	dummy1 = np.mean(np.reshape(body[body != 0.], [-1]))
	dummy2 = np.std(np.reshape(body[body != 0.], [-1]))

	body[body != 0.] -= dummy1
	body /= dummy2

	body[body != 0.] -= np.amin(np.reshape(body, [-1]))



	dummy3 = np.mean(np.reshape(body[body != 0.], [-1]))
	dummy4 = np.std(np.reshape(body[body != 0.], [-1]))

	dummy3 = np.round(dummy3, 2)
	dummy4 = np.round(dummy4, 2)

	# plt.figure()
	# plt.subplot(121)
	# plt.hist(np.reshape(body[body != 0.], [-1]), bins=40)
	# plt.title('mean:' + str(dummy3) + ' / std:' + str(dummy4))
	# plt.subplot(122)
	# # plt.imshow(np.mean(body[body != 0.], axis=1), aspect=8)
	# plt.imshow(np.true_divide(body.sum(1),(body!=0).sum(1)), aspect=8)
	# plt.title(str(Session_ID))
	# plt.show()



	if shrinker > 1:
		body = body[::shrinker,::shrinker,::shrinker]

	return body

def random_padder(scan_array, added_zero_layers=10, random=True, noise=True):

	if random:

		added_at_base = np.random.randint(0,added_zero_layers,3)
		added_at_end = added_zero_layers * np.ones(3) - added_at_base
		added_at_end = np.int32(added_at_end)

		if np.random.randint(0,5) != 0:

			random_flip = np.random.randint(0,3)
			scan_array = np.flip(scan_array, axis=random_flip)

			if np.random.randint(0,1) != 0:
				random_flip = np.random.randint(0,3)
				scan_array = np.flip(scan_array, axis=random_flip)

	else:

		added_at_base = added_zero_layers * .5 * np.ones(3)
		added_at_base = np.int32(added_at_base)
		added_at_end = added_zero_layers * np.ones(3) - added_at_base
		added_at_end = np.int32(added_at_end)

	if noise:

		scan_array[scan_array > 0.] += np.random.normal(0,.001,scan_array.shape)[scan_array > 0.]


	# 1st axis
	scan_array = np.concatenate((
			np.zeros((added_at_base[0], scan_array.shape[1], scan_array.shape[2])),
			scan_array,
			np.zeros((added_at_end[0], scan_array.shape[1], scan_array.shape[2]))),
		axis=0)

	# 2nd axis
	scan_array = np.concatenate((
			np.zeros((scan_array.shape[0], added_at_base[1], scan_array.shape[2])),
			scan_array,
			np.zeros((scan_array.shape[0], added_at_end[1], scan_array.shape[2]))),
		axis=1)

	# 3rd axis
	scan_array = np.concatenate((
			np.zeros((scan_array.shape[0], scan_array.shape[1], added_at_base[2])),
			scan_array,
			np.zeros((scan_array.shape[0], scan_array.shape[1], added_at_end[2]))),
		axis=2)


	return scan_array

def load_training_data(batch_index, training_batch_labels):

	body_array = load_body_array(training_batch_labels[batch_index[0],1])
	body_array = random_padder(body_array)

	batch_data = np.empty([len(batch_index), body_array.shape[0], body_array.shape[1], body_array.shape[2]])

	for i in range(len(batch_index)):
		
		body_array = load_body_array(training_batch_labels[batch_index[i],1])

		body_array = random_padder(body_array)

		if i == 0: 
			# batch_data = np.empty([
			# 	len(batch_index),
			# 	body_array.shape[0], 
			# 	body_array.shape[1], 
			# 	body_array.shape[2]])

			batch_labels = np.zeros([len(batch_index), len(use_cols)])

		batch_data[i,:,:,:] = body_array

		batch_labels[i,:] = training_batch_labels[batch_index[i],use_cols]

	batch_data_shaped = np.reshape(batch_data, [
		len(batch_index),
		body_array.shape[0], 
		body_array.shape[1], 
		body_array.shape[2],
		1])

	return batch_data_shaped, batch_labels

def load_validation_data():

	datadummy1 = np.load('validation_batch_data.npy')
	datadummy2 = np.load('validation_batch_labels.npy')

	return datadummy1, datadummy2

def load_test_data():

	datadummy1 = np.load('test_batch_data.npy')
	datadummy2 = np.load('test_batch_labels.npy')

	return datadummy1, datadummy2

def Preprocessing(directories=directories, figures=figures, num_bins_cont=8):

	Labels1_FileName = directories[0] + '/meta/labels.csv'
	Labels2_FileName = directories[1] + '/meta/labels.csv'
	Labels1, Labels2 = [], []

	with open(Labels1_FileName) as csv_file:
	    csv_reader = csv.reader(csv_file)
	    line_count = 0
	    for row in csv_reader:
	        Labels1.append(row)
	with open(Labels2_FileName) as csv_file:
	    csv_reader = csv.reader(csv_file)
	    line_count = 0
	    for row in csv_reader:
	        Labels2.append(row)
	if outprint:
		print()
		print()
		print('collected labels:')
		for name in range(np.asarray(Labels1).shape[1]):
			if name < 10: print(name, '.......', np.asarray(Labels1)[0, name])
			else: print(name, '......', np.asarray(Labels1)[0, name])
		print()
		print()

	Labels1 = np.asarray(Labels1)[1:,:]
	Labels2 = np.asarray(Labels2)[1:,:]
	Labels = np.concatenate((Labels1, Labels2))

	Labels[:,14][Labels[:,14] == 'yes'] = 1.
	Labels[:,14][Labels[:,14] == 'no'] = 0.
	Labels[:,15][Labels[:,14] == 'NA'] = Labels[:,14][Labels[:,14] == 'NA']
	Labels[:,15][Labels[:,15] == 'yes'] = 1.
	Labels[:,15][Labels[:,15] == 'no'] = 0.
	Labels[Labels == 'NA'] = 0.

	num_labels_init = Labels.shape[0]

	FileList = os.listdir(directories[2])
	FileList.pop(int(np.asarray(np.where(FileList[:5] != 'scan_'))))

	newfilelist = []
	for i in range(len(FileList)):
		newfilelist.append(FileList[i][5:-4])
	index_keep_fl = np.intersect1d(newfilelist, Labels[:,1])
	index_keep_fl2 = []
	for i in range(Labels.shape[0]): #N/A removal
		if Labels[i,1] in index_keep_fl:
			if Labels[i,1] not in ['1942', '2015', '1651', '1998', 'B00033', '986', '569', '91', '2113', '431', '249', '1558', 'B0028', '2345', '31', 'B00151', '1335', '496', '269', '4', '1249', '202', '2027', 'B00116', 'B00228', '3485', '835', '119', '1493', '1992', '3291', 'B00149']:
				if Labels[i,13] != 0. and Labels[i,17] !=0.:
					index_keep_fl2.append(i)
	Labels = Labels[index_keep_fl2, :]

	if figures:

		plt.figure(num='distribution of continuous features original', figsize=(30, 20))
		plot_cols = [4, 13, 6, 7, 8, 9, 10, 11, 12, 17]
		plot_titles = ['AGE', 'hb1ac', 'weight', 'BMI', 'fat_percentage_bioimpedance', 'insulin_sensitivity', 'total_adipose_tissue_liter', 'visceral_adipose_tissue_liter', 'adipose_tissue_upper_limbs_liter', 'DI']
		for which_plot in range(len(plot_cols)):
			plt.subplot(2,5,which_plot+1)
			plt.title(plot_titles[which_plot])
			plt.hist(np.float32(Labels[:,plot_cols[which_plot]]), bins=num_bins_cont, density=True, histtype='bar', color='r')
			plt.grid()
		plt.savefig(Target_dir + 'distribution_original.png', bbox_inches='tight')
		plt.close()




	pre_diab_vec = []
	for ii in range(Labels.shape[0]):

		if use_pre_diabetes == True:

			if np.float(Labels[ii,13]) >= 5.7:
				pre_diab_vec.append(1.)
			else:
				pre_diab_vec.append(0)

		if use_pre_diabetes == False:

			if np.float(Labels[ii,13]) >= 6.5:
				pre_diab_vec.append(1.)
			else:
				pre_diab_vec.append(0)


	pre_diab_vec = np.reshape(np.asarray(pre_diab_vec), [-1])

	print('ratio of diabetes/ pre-diabetes positives: ', np.mean(np.squeeze(pre_diab_vec)))

	Labels[:,15] = pre_diab_vec



	for feat in range(Labels.shape[1]): #outlier removal

		if feat == 0:
			reconstruction_scale = []


		if feat in [9, 17]: #drop zeros for DI and insulin_sensitivity
			index = np.asarray(np.where(np.reshape(np.float32(Labels[:,feat]), [-1]) > 0.))
			Labels = np.reshape(Labels[index,:], [-1,Labels.shape[1]])


		if feat == 9: #insulin_sensitivity upper outliers
			for outlier in range(4):
				index = np.asarray(np.where(np.reshape(np.float32(Labels[:,feat]), [-1]) < np.amax(np.reshape(np.float32(Labels[:,feat]), [-1]))))
				Labels = np.reshape(Labels[index,:], [-1,Labels.shape[1]])


		if feat == 5: #height lower outliers
			for outlier in range(2):
				index = np.asarray(np.where(np.reshape(np.float32(Labels[:,feat]), [-1]) > np.amin(np.reshape(np.float32(Labels[:,feat]), [-1]))))
				Labels = np.reshape(Labels[index,:], [-1,Labels.shape[1]])


		if feat == 13: #hb1ac lower outliers
			for outlier in range(6):
				index = np.asarray(np.where(np.reshape(np.float32(Labels[:,feat]), [-1]) > np.amin(np.reshape(np.float32(Labels[:,feat]), [-1]))))
				Labels = np.reshape(Labels[index,:], [-1,Labels.shape[1]])


		if feat in [6, 7, 8, 10, 12]: #upper outliers: weight/BMI/fat_percentage_bioimpedance/total_adipose_tissue_liter/adipose_tissue_upper_limbs_liter
			for outlier in range(2):
				index = np.asarray(np.where(np.reshape(np.float32(Labels[:,feat]), [-1]) < np.amax(np.reshape(np.float32(Labels[:,feat]), [-1]))))
				Labels = np.reshape(Labels[index,:], [-1,Labels.shape[1]])


		if feat == 24: #DI upper outliers
			for outlier in range(20):
				index = np.asarray(np.where(np.reshape(np.float32(Labels[:,feat]), [-1]) < np.amax(np.reshape(np.float32(Labels[:,feat]), [-1]))))
				Labels = np.reshape(Labels[index,:], [-1,Labels.shape[1]])

			
		if feat in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17]: #normalize [0,1] all continuous features
			Labels[:,feat] = np.reshape(np.float32(Labels[:,feat]), [-1]) - np.amin(np.reshape(np.float32(Labels[:,feat]), [-1]))
			for i in range(Labels.shape[0]):
				if np.float32(Labels[i,feat]) < .0001: Labels[i,feat] = .0001
			reconstruction_scale.append(np.amax(np.reshape(np.float32(Labels[:,feat]), [-1])))
			Labels[:,feat] = np.reshape(np.float32(Labels[:,feat]), [-1]) / np.amax(np.reshape(np.float32(Labels[:,feat]), [-1]))
			# Labels[:,feat] = (np.reshape(np.float32(Labels[:,feat]), [-1]) - np.mean(np.reshape(np.float32(Labels[:,feat]), [-1])))/ np.std(np.reshape(np.float32(Labels[:,feat]), [-1]))

		if feat not in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17]:
			reconstruction_scale.append(1)


	reconstruction_scale = np.reshape(np.asarray(reconstruction_scale), [-1])
	print()
	print()
	print('reconstruction_scale')
	print(reconstruction_scale.shape)
	print(reconstruction_scale)
	print()
	print()



	# diab_patient_cache = []
	# for patient in range(Labels.shape[0]):
	# 	if np.float(Labels[patient,14]) > 0.:
	# 		diab_patient_cache.append(patient)
	# Labels = np.concatenate((Labels, Labels[diab_patient_cache,:]), axis=0)


	ins_hist_vals, ins_bin_edges = np.histogram(np.float32(Labels[:,9]), num_bins_cont)
	binned_ins = np.reshape(np.digitize(np.float32(Labels[:,9]), ins_bin_edges), [-1,1])

	diab_all_hist_vals, diab_all_bin_edges = np.histogram(np.float32(Labels[:,15]), 2)
	binned_diab_all = np.reshape(np.digitize(np.float32(Labels[:,15]), diab_all_bin_edges), [-1,1])

	DI_hist_vals, DI_bin_edges = np.histogram(np.float32(Labels[:,17]), num_bins_cont)
	binned_DI = np.reshape(np.digitize(np.float32(Labels[:,17]), DI_bin_edges), [-1,1])

	stratified_y = np.concatenate((binned_ins, binned_diab_all, binned_DI), axis=-1)

	uni, uni1, uni2, uni3 = np.unique(stratified_y, return_index=True, return_inverse=True, return_counts=True, axis=0)

	if 1 in uni3:

		index_del = np.reshape(np.asarray(np.where(uni3 == 1)), [-1])
		num_solos = len(index_del)
		index_del2 = uni1[index_del]

		defect_cache = np.flip(np.sort(np.asarray(index_del2)), axis=-1)

		for i in range(num_solos):
			Labels = np.concatenate((Labels[:defect_cache[i],:], Labels[defect_cache[i]+1:,:]), axis=0)
			stratified_y = np.concatenate((stratified_y[:defect_cache[i],:], stratified_y[defect_cache[i]+1:,:]), axis=0)

	Labels = np.asarray(Labels)


	rho_spear, p_spear = spearmanr(Labels[:,corr_cols])

	print('-----------------------------', 'rho_spear')
	print('-----------------------------', rho_spear.shape)
	print('-----------------------------', rho_spear)
	print()
	print('-----------------------------', 'p_spear')
	print('-----------------------------', p_spear.shape)
	print('-----------------------------', p_spear)

	plt.figure('spearman correlation', figsize=(24,10))

	plt.subplot(121)
	plt.title('Spearman Correlation Matrix')
	plt.imshow(rho_spear)
	plt.xticks(np.arange(rho_spear.shape[0]), corr_col_names)
	plt.yticks(np.arange(rho_spear.shape[0]), corr_col_names)
	plt.colorbar()

	plt.subplot(122)
	plt.title('p-value matrix')
	plt.imshow(p_spear)
	plt.xticks(np.arange(rho_spear.shape[0]), corr_col_names)
	plt.yticks(np.arange(rho_spear.shape[0]), corr_col_names)
	plt.colorbar()

	plt.show()


	Labels = np.concatenate((Labels, np.reshape(np.arange(Labels.shape[0]), [-1,1])), axis=1)
	Labels = np.asarray(Labels)

	np.save('lookup', Labels)


	stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=int(test_ratio*Labels.shape[0]/batch_size)*batch_size, train_size=None, random_state=420)

	for train_index, test_index in stratified_splitter.split(np.zeros(Labels.shape[0]), stratified_y):
		stratified_index_training = train_index
		stratified_index_validation = test_index


	validation_batch_labels0 = Labels[stratified_index_validation,:]
	training_batch_labels = Labels[stratified_index_training,:]

	new_labels = np.concatenate((validation_batch_labels0, training_batch_labels), axis=0)








	stratified_y = stratified_y[stratified_index_validation,:]


	uni, uni1, uni2, uni3 = np.unique(stratified_y, return_index=True, return_inverse=True, return_counts=True, axis=0)

	if 1 in uni3:

		index_del = np.reshape(np.asarray(np.where(uni3 == 1)), [-1])
		num_solos = len(index_del)
		index_del2 = uni1[index_del]

		defect_cache = np.flip(np.sort(np.asarray(index_del2)), axis=-1)

		for i in range(num_solos):
			stratified_y = np.concatenate((stratified_y[:defect_cache[i],:], stratified_y[defect_cache[i]+1:,:]), axis=0)



	stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=int(.3*stratified_y.shape[0]), train_size=None, random_state=420)

	for val_index, test_index in stratified_splitter.split(np.zeros(stratified_y.shape[0]), stratified_y):
		stratified_index_validation = val_index
		stratified_index_test = test_index


	print(stratified_index_training.shape)
	print(stratified_index_validation.shape)
	print(stratified_index_test.shape)

	validation_batch_labels = validation_batch_labels0[stratified_index_validation,:]
	test_batch_labels = validation_batch_labels0[stratified_index_test,:]





	if outprint:
		for space in range(2): print()
		print('total number of loaded body arrays.........', new_labels.shape[0], '/', num_labels_init)
		print('training samples...........................', training_batch_labels.shape)
		print('validation samples.........................', validation_batch_labels.shape)
		print('test samples...............................', test_batch_labels.shape)
		for space in range(4): print()


	if do_sklearn:


		# use_as_labels = test_batch_labels
		use_as_labels = validation_batch_labels


		#################################################################################################
		#################################################################################################
		###################################### insulin_sensitivity ######################################
		#################################################################################################
		#################################################################################################


		print('benchmark scores from features: training set')
		print('--------------------------------------------')
		print('rmse of normalised insulin_sensitivity')
		print()


		model_sklean = LinearRegression()
		use_as_x, use_as_y = [7], 9
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1,1])).predict(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))
		# print('LinearRegression(bmi)------------------------', np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3))
		sklearn_score_cache = np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3)
		expvar_skl_cache = np.round(compute_explained_variance(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1])), 2)


		model_sklean = LinearRegression()
		use_as_x, use_as_y = [3,4,5,6,7], 9 #,11,12
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1,1])).predict(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))
		print('LinearRegression [bmi/all]')
		print('.........................RMSE: ', sklearn_score_cache, '/', np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3))
		print('................explained_var: ', expvar_skl_cache, '/', np.round(compute_explained_variance(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1])), 2))

		model_sklean = RandomForestRegressor()
		use_as_x, use_as_y = [7], 9
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1])).predict(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))
		# print('RandomForestRegressor(bmi)-------------------', np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3))
		sklearn_score_cache = np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3)
		expvar_skl_cache = np.round(compute_explained_variance(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1])), 2)

		model_sklean = RandomForestRegressor()	
		use_as_x, use_as_y = [3,4,5,6,7], 9
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1])).predict(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))
		print('RandomForestRegressor [bmi/all]')
		print('.........................RMSE: ', sklearn_score_cache, '/', np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3))
		print('................explained_var: ', expvar_skl_cache, '/', np.round(compute_explained_variance(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1])), 2))


		model_sklean = SVR(kernel='rbf', degree=3)
		use_as_x, use_as_y = [7], 9
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1])).predict(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))
		# print('SupportVectorRegressor(bmi)------------------', np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3))
		sklearn_score_cache = np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3)
		expvar_skl_cache = np.round(compute_explained_variance(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1])), 2)

		model_sklean = SVR(kernel='rbf', degree=3)
		use_as_x, use_as_y = [3,4,5,6,7], 9
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1])).predict(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))
		print('SupportVectorRegressor [bmi/all]')
		print('.........................RMSE: ', sklearn_score_cache, '/', np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3))
		print('................explained_var: ', expvar_skl_cache, '/', np.round(compute_explained_variance(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1])), 2))


		#################################################################################################
		#################################################################################################
		########################################### DI index ############################################
		#################################################################################################
		#################################################################################################


		print()
		print()
		print('rmse of normalised DI index')
		print()



		model_sklean = LinearRegression()
		use_as_x, use_as_y = [7], 17
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1,1])).predict(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))
		# print('LinearRegression(bmi)------------------------', np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3))
		sklearn_score_cache = np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3)
		expvar_skl_cache = np.round(compute_explained_variance(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1])), 2)

		model_sklean = LinearRegression()
		use_as_x, use_as_y = [3,4,5,6,7], 17
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1,1])).predict(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))
		print('LinearRegression [bmi/all]')
		print('.........................RMSE: ', sklearn_score_cache, '/', np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3))
		print('................explained_var: ', expvar_skl_cache, '/', np.round(compute_explained_variance(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1])), 2))


		model_sklean = RandomForestRegressor()	
		use_as_x, use_as_y = [7], 17
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1])).predict(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))
		# print('RandomForestRegressor(bmi)-------------------', np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3))
		sklearn_score_cache = np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3)
		expvar_skl_cache = np.round(compute_explained_variance(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1])), 2)

		model_sklean = RandomForestRegressor()	
		use_as_x, use_as_y = [3,4,5,6,7], 17
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1])).predict(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))
		print('RandomForestRegressor [bmi/all]')
		print('.........................RMSE: ', sklearn_score_cache, '/', np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3))
		print('................explained_var: ', expvar_skl_cache, '/', np.round(compute_explained_variance(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1])), 2))
		
		model_sklean = SVR(kernel='rbf', degree=3)
		use_as_x, use_as_y = [7], 17
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1])).predict(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))
		# print('SupportVectorRegressor(bmi)------------------', np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3))
		sklearn_score_cache = np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3)
		expvar_skl_cache = np.round(compute_explained_variance(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1])), 2)

		model_sklean = SVR(kernel='rbf', degree=3)
		use_as_x, use_as_y = [3,4,5,6,7], 17
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1])).predict(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))
		print('SupportVectorRegressor [bmi/all]')
		print('.........................RMSE: ', sklearn_score_cache, '/', np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3))
		print('................explained_var: ', expvar_skl_cache, '/', np.round(compute_explained_variance(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1])), 2))



		#################################################################################################
		#################################################################################################
		###################################### hb1ac regression #########################################
		#################################################################################################
		#################################################################################################


		print()
		print()
		print('rmse of normalised Hb1Ac')
		print()





		model_sklean = LinearRegression()
		use_as_x, use_as_y = [7], 13
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1,1])).predict(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))
		sklearn_score_cache = np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3)
		error_rates = []
		error_rates.append(roc_Hb1Ac(y_pred, use_as_labels[:,15]))
		error_rates_names = []
		error_rates_names.append('LinReg_BMI')
		# area1, accuracies = roc_Hb1Ac(y_pred, use_as_labels[:,15])
		# area_cache = np.round(-100. * np.trapz(area1[1], x=area1[0], axis=0))
		area_cache = np.round(100*compute_auc(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2)
		expvar_skl_cache = np.round(compute_explained_variance(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1])), 2)
		precision_skl_cache = np.round(compute_precision_recall(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2)


		model_sklean = LinearRegression()
		use_as_x, use_as_y = [3,4,5,6,7], 13
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1,1])).predict(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))
		error_rates.append(roc_Hb1Ac(y_pred, use_as_labels[:,15]))
		error_rates_names.append('LinReg_all')
		area1, accuracies = roc_Hb1Ac(y_pred, use_as_labels[:,15])

		print('LinearRegression [bmi/all]')
		print('.........................RMSE: ', sklearn_score_cache, '/', np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3))
		print('................explained_var: ', expvar_skl_cache, '/', np.round(compute_explained_variance(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1])), 2))
		print('......................AUC_ROC: ', area_cache, '/', np.round(100*compute_auc(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2))
		print('....................precision: ', precision_skl_cache, '/', np.round(compute_precision_recall(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2))


		model_sklean = RandomForestRegressor()	
		use_as_x, use_as_y = [7], 13
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1])).predict(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))
		sklearn_score_cache = np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3)
		error_rates.append(roc_Hb1Ac(y_pred, use_as_labels[:,15]))
		error_rates_names.append('RandForest_BMI')
		# area1, accuracies = roc_Hb1Ac(y_pred, use_as_labels[:,15])
		# area_cache = np.round(-100. * np.trapz(area1[1], x=area1[0], axis=0))
		area_cache = np.round(100*compute_auc(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2)
		expvar_skl_cache = np.round(compute_explained_variance(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1])), 2)
		precision_skl_cache = np.round(compute_precision_recall(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2)


		model_sklean = RandomForestRegressor()	
		use_as_x, use_as_y = [3,4,5,6,7], 13
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1])).predict(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))
		error_rates.append(roc_Hb1Ac(y_pred, use_as_labels[:,15]))
		error_rates_names.append('RandForest_all')
		area1, accuracies = roc_Hb1Ac(y_pred, use_as_labels[:,15])

		print('RandomForestRegressor [bmi/all]')
		print('.........................RMSE: ', sklearn_score_cache, '/', np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3))
		print('................explained_var: ', expvar_skl_cache, '/', np.round(compute_explained_variance(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1])), 2))
		print('......................AUC_ROC: ', area_cache, '/', np.round(100*compute_auc(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2))
		print('....................precision: ', precision_skl_cache, '/', np.round(compute_precision_recall(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2))


		model_sklean = SVR(kernel='rbf', degree=3)
		use_as_x, use_as_y = [7], 13
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1])).predict(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))
		sklearn_score_cache = np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3)
		error_rates.append(roc_Hb1Ac(y_pred, use_as_labels[:,15]))
		error_rates_names.append('SVR_BMI')
		# area1, accuracies = roc_Hb1Ac(y_pred, use_as_labels[:,15])
		# area_cache = np.round(-100. * np.trapz(area1[1], x=area1[0], axis=0))
		area_cache = np.round(100*compute_auc(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2)
		expvar_skl_cache = np.round(compute_explained_variance(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1])), 2)
		precision_skl_cache = np.round(compute_precision_recall(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2)


		model_sklean = SVR(kernel='rbf', degree=3)
		use_as_x, use_as_y = [3,4,5,6,7], 13
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1])).predict(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))
		error_rates.append(roc_Hb1Ac(y_pred, use_as_labels[:,15]))
		error_rates_names.append('RandomForest_all')
		area1, accuracies = roc_Hb1Ac(y_pred, use_as_labels[:,15])

		print('SupportVectorRegressor [bmi/all]')
		print('.........................RMSE: ', sklearn_score_cache, '/', np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3))
		print('................explained_var: ', expvar_skl_cache, '/', np.round(compute_explained_variance(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1])), 2))
		print('......................AUC_ROC: ', area_cache, '/', np.round(100*compute_auc(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2))
		print('....................precision: ', precision_skl_cache, '/', np.round(compute_precision_recall(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2))






		#################################################################################################
		#################################################################################################
		################################## diabetes classification ######################################
		#################################################################################################
		#################################################################################################


		print()
		print()
		print('diabetes classification roc')
		print()




		model_sklean = SVC(kernel='rbf', degree=3, probability=True)
		use_as_x, use_as_y = [7], 15
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1,1])).predict_proba(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))[:,1]
		sklearn_score_cache = np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3)
		# error_rates = []
		error_rates.append(roc_Hb1Ac(y_pred, use_as_labels[:,14]))
		# error_rates_names = []
		error_rates_names.append('SVM_BMI')
		# area1, accuracies = roc_Hb1Ac(y_pred, use_as_labels[:,14])
		# area_cache = np.round(-100. * np.trapz(area1[1], x=area1[0], axis=0))
		area_cache = np.round(100*compute_auc(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2)
		precision_skl_cache = np.round(compute_precision_recall(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2)


		model_sklean = SVC(kernel='rbf', degree=3, probability=True)
		use_as_x, use_as_y = [3,4,5,6,7], 15
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1,1])).predict_proba(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))[:,1]
		error_rates.append(roc_Hb1Ac(y_pred, use_as_labels[:,14]))
		error_rates_names.append('SVM_all')
		area1, accuracies = roc_Hb1Ac(y_pred, use_as_labels[:,14])

		print('SupportVectorMachines [bmi/all]')
		print('.........................RMSE: ', sklearn_score_cache, '/', np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3))
		print('......................AUC_ROC: ', area_cache, '/', np.round(100*compute_auc(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2))
		print('....................precision: ', precision_skl_cache, '/', np.round(compute_precision_recall(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2))


		model_sklean = KNeighborsClassifier(n_neighbors=5)
		use_as_x, use_as_y = [7], 15
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1])).predict_proba(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))[:,1]
		sklearn_score_cache = np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3)
		error_rates.append(roc_Hb1Ac(y_pred, use_as_labels[:,14]))
		error_rates_names.append('KNN_BMI')
		# area1, accuracies = roc_Hb1Ac(y_pred, use_as_labels[:,14])
		# area_cache = np.round(-100. * np.trapz(area1[1], x=area1[0], axis=0))
		area_cache = np.round(100*compute_auc(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2)
		precision_skl_cache = np.round(compute_precision_recall(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2)


		model_sklean = KNeighborsClassifier(n_neighbors=5)
		use_as_x, use_as_y = [3,4,5,6,7], 15
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1])).predict_proba(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))[:,1]
		error_rates.append(roc_Hb1Ac(y_pred, use_as_labels[:,14]))
		error_rates_names.append('KNN_all')
		area1, accuracies = roc_Hb1Ac(y_pred, use_as_labels[:,14])

		print('KNeighborsClassifier [bmi/all]')
		print('.........................RMSE: ', sklearn_score_cache, '/', np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3))
		print('......................AUC_ROC: ', area_cache, '/', np.round(100*compute_auc(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2))
		print('....................precision: ', precision_skl_cache, '/', np.round(compute_precision_recall(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2))



		model_sklean = RandomForestClassifier()
		use_as_x, use_as_y = [7], 15
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1])).predict_proba(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))[:,1]
		sklearn_score_cache = np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3)
		error_rates.append(roc_Hb1Ac(y_pred, use_as_labels[:,14]))
		error_rates_names.append('RandForestC_BMI')
		# area1, accuracies = roc_Hb1Ac(y_pred, use_as_labels[:,14])
		# area_cache = np.round(-100. * np.trapz(area1[1], x=area1[0], axis=0))
		area_cache = np.round(100*compute_auc(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2)
		precision_skl_cache = np.round(compute_precision_recall(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2)


		model_sklean = RandomForestClassifier()
		use_as_x, use_as_y = [3,4,5,6,7], 15
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1])).predict_proba(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))[:,1]
		error_rates.append(roc_Hb1Ac(y_pred, use_as_labels[:,14]))
		error_rates_names.append('RandForestC_all')
		area1, accuracies = roc_Hb1Ac(y_pred, use_as_labels[:,14])

		print('RandomForestClassifier [bmi/all]')
		print('.........................RMSE: ', sklearn_score_cache, '/', np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3))
		print('......................AUC_ROC: ', area_cache, '/', np.round(100*compute_auc(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2))
		print('....................precision: ', precision_skl_cache, '/', np.round(compute_precision_recall(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2))







		##############################################################################################################################################################
		##############################################################################################################################################################
		##############################################################################################################################################################
		##############################################################################################################################################################
		##############################################################################################################################################################
		##############################################################################################################################################################
		##############################################################################################################################################################






		use_as_labels = test_batch_labels
		# use_as_labels = validation_batch_labels


		#################################################################################################
		#################################################################################################
		###################################### insulin_sensitivity ######################################
		#################################################################################################
		#################################################################################################

		print()
		print('benchmark scores from features: test set')
		print('--------------------------------------------')
		print('rmse of normalised insulin_sensitivity')
		print()



		model_sklean = LinearRegression()
		use_as_x, use_as_y = [7], 9
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1,1])).predict(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))
		# print('LinearRegression(bmi)------------------------', np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3))
		sklearn_score_cache = np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3)
		expvar_skl_cache = np.round(compute_explained_variance(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1])), 2)


		model_sklean = LinearRegression()
		use_as_x, use_as_y = [3,4,5,6,7], 9 #,11,12
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1,1])).predict(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))
		print('LinearRegression [bmi/all]')
		print('.........................RMSE: ', sklearn_score_cache, '/', np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3))
		print('................explained_var: ', expvar_skl_cache, '/', np.round(compute_explained_variance(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1])), 2))

		model_sklean = RandomForestRegressor()
		use_as_x, use_as_y = [7], 9
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1])).predict(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))
		# print('RandomForestRegressor(bmi)-------------------', np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3))
		sklearn_score_cache = np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3)
		expvar_skl_cache = np.round(compute_explained_variance(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1])), 2)

		model_sklean = RandomForestRegressor()	
		use_as_x, use_as_y = [3,4,5,6,7], 9
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1])).predict(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))
		print('RandomForestRegressor [bmi/all]')
		print('.........................RMSE: ', sklearn_score_cache, '/', np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3))
		print('................explained_var: ', expvar_skl_cache, '/', np.round(compute_explained_variance(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1])), 2))


		model_sklean = SVR(kernel='rbf', degree=3)
		use_as_x, use_as_y = [7], 9
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1])).predict(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))
		# print('SupportVectorRegressor(bmi)------------------', np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3))
		sklearn_score_cache = np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3)
		expvar_skl_cache = np.round(compute_explained_variance(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1])), 2)

		model_sklean = SVR(kernel='rbf', degree=3)
		use_as_x, use_as_y = [3,4,5,6,7], 9
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1])).predict(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))
		print('SupportVectorRegressor [bmi/all]')
		print('.........................RMSE: ', sklearn_score_cache, '/', np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3))
		print('................explained_var: ', expvar_skl_cache, '/', np.round(compute_explained_variance(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1])), 2))


		#################################################################################################
		#################################################################################################
		########################################### DI index ############################################
		#################################################################################################
		#################################################################################################


		print()
		print()
		print('rmse of normalised DI index')
		print()



		model_sklean = LinearRegression()
		use_as_x, use_as_y = [7], 17
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1,1])).predict(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))
		# print('LinearRegression(bmi)------------------------', np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3))
		sklearn_score_cache = np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3)
		expvar_skl_cache = np.round(compute_explained_variance(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1])), 2)

		model_sklean = LinearRegression()
		use_as_x, use_as_y = [3,4,5,6,7], 17
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1,1])).predict(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))
		print('LinearRegression [bmi/all]')
		print('.........................RMSE: ', sklearn_score_cache, '/', np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3))
		print('................explained_var: ', expvar_skl_cache, '/', np.round(compute_explained_variance(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1])), 2))


		model_sklean = RandomForestRegressor()	
		use_as_x, use_as_y = [7], 17
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1])).predict(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))
		# print('RandomForestRegressor(bmi)-------------------', np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3))
		sklearn_score_cache = np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3)
		expvar_skl_cache = np.round(compute_explained_variance(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1])), 2)

		model_sklean = RandomForestRegressor()	
		use_as_x, use_as_y = [3,4,5,6,7], 17
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1])).predict(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))
		print('RandomForestRegressor [bmi/all]')
		print('.........................RMSE: ', sklearn_score_cache, '/', np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3))
		print('................explained_var: ', expvar_skl_cache, '/', np.round(compute_explained_variance(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1])), 2))
		
		model_sklean = SVR(kernel='rbf', degree=3)
		use_as_x, use_as_y = [7], 17
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1])).predict(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))
		# print('SupportVectorRegressor(bmi)------------------', np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3))
		sklearn_score_cache = np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3)
		expvar_skl_cache = np.round(compute_explained_variance(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1])), 2)

		model_sklean = SVR(kernel='rbf', degree=3)
		use_as_x, use_as_y = [3,4,5,6,7], 17
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1])).predict(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))
		print('SupportVectorRegressor [bmi/all]')
		print('.........................RMSE: ', sklearn_score_cache, '/', np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3))
		print('................explained_var: ', expvar_skl_cache, '/', np.round(compute_explained_variance(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1])), 2))



		#################################################################################################
		#################################################################################################
		###################################### hb1ac regression #########################################
		#################################################################################################
		#################################################################################################


		print()
		print()
		print('rmse of normalised Hb1Ac')
		print()





		model_sklean = LinearRegression()
		use_as_x, use_as_y = [7], 13
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1,1])).predict(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))
		sklearn_score_cache = np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3)
		error_rates = []
		error_rates.append(roc_Hb1Ac(y_pred, use_as_labels[:,15]))
		error_rates_names = []
		error_rates_names.append('LinReg_BMI')
		# area1, accuracies = roc_Hb1Ac(y_pred, use_as_labels[:,15])
		# area_cache = np.round(-100. * np.trapz(area1[1], x=area1[0], axis=0))
		area_cache = np.round(100*compute_auc(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2)
		expvar_skl_cache = np.round(compute_explained_variance(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1])), 2)
		precision_skl_cache = np.round(compute_precision_recall(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2)


		model_sklean = LinearRegression()
		use_as_x, use_as_y = [3,4,5,6,7], 13
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1,1])).predict(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))
		error_rates.append(roc_Hb1Ac(y_pred, use_as_labels[:,15]))
		error_rates_names.append('LinReg_all')
		area1, accuracies = roc_Hb1Ac(y_pred, use_as_labels[:,15])

		print('LinearRegression [bmi/all]')
		print('.........................RMSE: ', sklearn_score_cache, '/', np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3))
		print('................explained_var: ', expvar_skl_cache, '/', np.round(compute_explained_variance(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1])), 2))
		print('......................AUC_ROC: ', area_cache, '/', np.round(100*compute_auc(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2))
		print('....................precision: ', precision_skl_cache, '/', np.round(compute_precision_recall(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2))


		model_sklean = RandomForestRegressor()	
		use_as_x, use_as_y = [7], 13
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1])).predict(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))
		sklearn_score_cache = np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3)
		error_rates.append(roc_Hb1Ac(y_pred, use_as_labels[:,15]))
		error_rates_names.append('RandForest_BMI')
		# area1, accuracies = roc_Hb1Ac(y_pred, use_as_labels[:,15])
		# area_cache = np.round(-100. * np.trapz(area1[1], x=area1[0], axis=0))
		area_cache = np.round(100*compute_auc(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2)
		expvar_skl_cache = np.round(compute_explained_variance(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1])), 2)
		precision_skl_cache = np.round(compute_precision_recall(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2)


		model_sklean = RandomForestRegressor()	
		use_as_x, use_as_y = [3,4,5,6,7], 13
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1])).predict(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))
		error_rates.append(roc_Hb1Ac(y_pred, use_as_labels[:,15]))
		error_rates_names.append('RandForest_all')
		area1, accuracies = roc_Hb1Ac(y_pred, use_as_labels[:,15])

		print('RandomForestRegressor [bmi/all]')
		print('.........................RMSE: ', sklearn_score_cache, '/', np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3))
		print('................explained_var: ', expvar_skl_cache, '/', np.round(compute_explained_variance(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1])), 2))
		print('......................AUC_ROC: ', area_cache, '/', np.round(100*compute_auc(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2))
		print('....................precision: ', precision_skl_cache, '/', np.round(compute_precision_recall(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2))


		model_sklean = SVR(kernel='rbf', degree=3)
		use_as_x, use_as_y = [7], 13
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1])).predict(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))
		sklearn_score_cache = np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3)
		error_rates.append(roc_Hb1Ac(y_pred, use_as_labels[:,15]))
		error_rates_names.append('SVR_BMI')
		# area1, accuracies = roc_Hb1Ac(y_pred, use_as_labels[:,15])
		# area_cache = np.round(-100. * np.trapz(area1[1], x=area1[0], axis=0))
		area_cache = np.round(100*compute_auc(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2)
		expvar_skl_cache = np.round(compute_explained_variance(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1])), 2)
		precision_skl_cache = np.round(compute_precision_recall(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2)


		model_sklean = SVR(kernel='rbf', degree=3)
		use_as_x, use_as_y = [3,4,5,6,7], 13
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1])).predict(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))
		error_rates.append(roc_Hb1Ac(y_pred, use_as_labels[:,15]))
		error_rates_names.append('RandomForest_all')
		area1, accuracies = roc_Hb1Ac(y_pred, use_as_labels[:,15])

		print('SupportVectorRegressor [bmi/all]')
		print('.........................RMSE: ', sklearn_score_cache, '/', np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3))
		print('................explained_var: ', expvar_skl_cache, '/', np.round(compute_explained_variance(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1])), 2))
		print('......................AUC_ROC: ', area_cache, '/', np.round(100*compute_auc(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2))
		print('....................precision: ', precision_skl_cache, '/', np.round(compute_precision_recall(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2))






		#################################################################################################
		#################################################################################################
		################################## diabetes classification ######################################
		#################################################################################################
		#################################################################################################


		print()
		print()
		print('diabetes classification roc')
		print()





		model_sklean = SVC(kernel='rbf', degree=3, probability=True)
		use_as_x, use_as_y = [7], 15
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1,1])).predict_proba(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))[:,1]
		sklearn_score_cache = np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3)
		# error_rates = []
		error_rates.append(roc_Hb1Ac(y_pred, use_as_labels[:,14]))
		# error_rates_names = []
		error_rates_names.append('SVM_BMI')
		# area1, accuracies = roc_Hb1Ac(y_pred, use_as_labels[:,14])
		# area_cache = np.round(-100. * np.trapz(area1[1], x=area1[0], axis=0))
		area_cache = np.round(100*compute_auc(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2)
		precision_skl_cache = np.round(compute_precision_recall(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2)


		model_sklean = SVC(kernel='rbf', degree=3, probability=True)
		use_as_x, use_as_y = [3,4,5,6,7], 15
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1,1])).predict_proba(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))[:,1]
		error_rates.append(roc_Hb1Ac(y_pred, use_as_labels[:,14]))
		error_rates_names.append('SVM_all')
		area1, accuracies = roc_Hb1Ac(y_pred, use_as_labels[:,14])

		print('SupportVectorMachines [bmi/all]')
		print('.........................RMSE: ', sklearn_score_cache, '/', np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3))
		print('......................AUC_ROC: ', area_cache, '/', np.round(100*compute_auc(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2))
		print('....................precision: ', precision_skl_cache, '/', np.round(compute_precision_recall(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2))


		model_sklean = KNeighborsClassifier(n_neighbors=5)
		use_as_x, use_as_y = [7], 15
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1])).predict_proba(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))[:,1]
		sklearn_score_cache = np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3)
		error_rates.append(roc_Hb1Ac(y_pred, use_as_labels[:,14]))
		error_rates_names.append('KNN_BMI')
		# area1, accuracies = roc_Hb1Ac(y_pred, use_as_labels[:,14])
		# area_cache = np.round(-100. * np.trapz(area1[1], x=area1[0], axis=0))
		area_cache = np.round(100*compute_auc(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2)
		precision_skl_cache = np.round(compute_precision_recall(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2)


		model_sklean = KNeighborsClassifier(n_neighbors=5)
		use_as_x, use_as_y = [3,4,5,6,7], 15
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1])).predict_proba(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))[:,1]
		error_rates.append(roc_Hb1Ac(y_pred, use_as_labels[:,14]))
		error_rates_names.append('KNN_all')
		area1, accuracies = roc_Hb1Ac(y_pred, use_as_labels[:,14])

		print('KNeighborsClassifier [bmi/all]')
		print('.........................RMSE: ', sklearn_score_cache, '/', np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3))
		print('......................AUC_ROC: ', area_cache, '/', np.round(100*compute_auc(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2))
		print('....................precision: ', precision_skl_cache, '/', np.round(compute_precision_recall(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2))



		model_sklean = RandomForestClassifier()
		use_as_x, use_as_y = [7], 15
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1])).predict_proba(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))[:,1]
		sklearn_score_cache = np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3)
		error_rates.append(roc_Hb1Ac(y_pred, use_as_labels[:,14]))
		error_rates_names.append('RandForestC_BMI')
		# area1, accuracies = roc_Hb1Ac(y_pred, use_as_labels[:,14])
		# area_cache = np.round(-100. * np.trapz(area1[1], x=area1[0], axis=0))
		area_cache = np.round(100*compute_auc(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2)
		precision_skl_cache = np.round(compute_precision_recall(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2)


		model_sklean = RandomForestClassifier()
		use_as_x, use_as_y = [3,4,5,6,7], 15
		y_pred = model_sklean.fit(np.reshape(np.float32(training_batch_labels[:,use_as_x]), [-1,len(use_as_x)]), np.reshape(np.float32(training_batch_labels[:,use_as_y]) , [-1])).predict_proba(np.reshape(np.float32(use_as_labels[:,use_as_x]), [-1,len(use_as_x)]))[:,1]
		error_rates.append(roc_Hb1Ac(y_pred, use_as_labels[:,14]))
		error_rates_names.append('RandForestC_all')
		area1, accuracies = roc_Hb1Ac(y_pred, use_as_labels[:,14])

		print('RandomForestClassifier [bmi/all]')
		print('.........................RMSE: ', sklearn_score_cache, '/', np.round(np.sqrt(mean_squared_error(y_pred, np.reshape(np.float32(use_as_labels[:,use_as_y]) , [-1,1]))),3))
		print('......................AUC_ROC: ', area_cache, '/', np.round(100*compute_auc(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2))
		print('....................precision: ', precision_skl_cache, '/', np.round(compute_precision_recall(np.asarray(y_pred), np.reshape(np.asarray(use_as_labels)[:,15], [-1])), 2))


		sys.stdout.flush()






		##############################################################################################################################################################
		##############################################################################################################################################################
		##############################################################################################################################################################
		##############################################################################################################################################################
		##############################################################################################################################################################
		##############################################################################################################################################################
		##############################################################################################################################################################







		np.save(Target_dir + 'error_rates_sklearn', error_rates)
		np.save(Target_dir + 'error_rates_sklearn_names', error_rates_names)
		np.save(Target_dir + 'error_rates_labels', use_as_labels[:,15])
		for spacing in range(6): print()



	if figures:

		print('building figures')
		print()
		print()

		plt.figure(num='distribution of continuous features after preprocessing', figsize=(30, 20))
		plot_cols = [4, 13, 6, 7, 8, 9, 10, 11, 12, 17]
		plot_titles = ['AGE', 'hb1ac', 'weight', 'BMI', 'fat_percentage_bioimpedance', 'insulin_sensitivity', 'total_adipose_tissue_liter', 'visceral_adipose_tissue_liter', 'adipose_tissue_upper_limbs_liter', 'DI']
		for which_plot in range(len(plot_cols)):
			plt.subplot(2,5,which_plot+1)
			plt.title(plot_titles[which_plot])
			plt.hist(np.float32(Labels[:,plot_cols[which_plot]]), bins=num_bins_cont, histtype='bar', color='g')
			plt.grid()
		plt.savefig(Target_dir + 'distribution_processed.png', bbox_inches='tight')
		plt.close()

	if outprint: print('loading validation batch...')


	for i in range(len(stratified_index_validation)):
		
		body_array = load_body_array(validation_batch_labels[i,1])

		body_array = random_padder(body_array, random=False, noise=False)


		if i == 0: validation_batch_data = np.empty([len(stratified_index_validation),
													 body_array.shape[0], 
													 body_array.shape[1], 
													 body_array.shape[2]])

		validation_batch_data[i,:,:,:] = body_array


	validation_batch_data = np.reshape(validation_batch_data, [
		len(stratified_index_validation),
		body_array.shape[0], 
		body_array.shape[1], 
		body_array.shape[2],
		1])


	for i in range(len(stratified_index_test)):
		
		body_array = load_body_array(test_batch_labels[i,1])

		body_array = random_padder(body_array, random=False, noise=False)

		# plt.plot(np.reshape(np.mean(np.mean(body_array, axis=1), axis=1), [-1]))
		# plt.figure()
		# plt.subplot(221)
		# plt.imshow(np.mean(body_array, axis=1), aspect=8)
		# plt.title(str(validation_batch_labels[i,1]))
		# plt.subplot(222)
		# plt.imshow(np.mean(np.flip(body_array, axis=0), axis=1), aspect=8)
		# plt.title('flip0')
		# plt.subplot(223)
		# plt.imshow(np.mean(np.flip(body_array, axis=1), axis=1), aspect=8)
		# plt.title('flip1')
		# plt.subplot(224)
		# plt.imshow(np.mean(np.flip(body_array, axis=2), axis=1), aspect=8)
		# plt.title('flip2')
		# plt.show()


		if i == 0: test_batch_data = np.empty([len(stratified_index_test),
											   body_array.shape[0], 
											   body_array.shape[1], 
											   body_array.shape[2]])

		test_batch_data[i,:,:,:] = body_array


	test_batch_data = np.reshape(test_batch_data, [
		len(stratified_index_test),
		body_array.shape[0], 
		body_array.shape[1], 
		body_array.shape[2],
		1])


	body_array_shape = [body_array.shape[0], body_array.shape[1], body_array.shape[2]]

	if outprint: print('writing validation batch to disk...')
	np.save('validation_batch_data', validation_batch_data)
	np.save('validation_batch_labels', validation_batch_labels[:,use_cols])

	if outprint: print('writing test batch to disk...')
	np.save('test_batch_data', test_batch_data)
	np.save('test_batch_labels', test_batch_labels[:,use_cols])


	return training_batch_labels, body_array_shape, stratified_index_validation.shape[0]

def make_folder(epoch, partition):

	if epoch < 10: 
		epoch_counter = str(00) + str(epoch)
	if epoch >= 10 and epoch < 100:
		epoch_counter = str(0) + str(epoch)
	if epoch >= 100:
		epoch_counter = str(epoch)
	
	if partition < 10: 
		partition_counter = str(0) + str(partition)
	else:
		partition_counter = str(partition)


	os.mkdir(Target_dir + 'val_epoch' + epoch_counter + '_' + partition_counter)

	return Target_dir + 'val_epoch' + epoch_counter + '_' + partition_counter

def compute_explained_variance(pred_vec, truth_vec):

	truth_vec = np.reshape(truth_vec, [-1])
	pred_vec = np.reshape(pred_vec, [-1])

	truth_floats = []
	pred_floats = []

	for number in range(len(truth_vec)):
		truth_floats.append(np.float(truth_vec[number]))
		pred_floats.append(np.float(pred_vec[number]))
	truth_floats = np.reshape(np.asarray(truth_floats), [-1])
	pred_floats = np.reshape(np.asarray(pred_floats), [-1])

	expl_var = explained_variance_score(truth_floats, pred_vec)

	return expl_var

def compute_precision_recall(pred_vec, truth_vec):

	truth_vec = np.reshape(truth_vec, [-1])
	pred_vec = np.reshape(pred_vec, [-1])

	truth_floats = []
	pred_floats = []

	for number in range(len(truth_vec)):
		truth_floats.append(np.float(truth_vec[number]))
		pred_floats.append(np.float(pred_vec[number]))
	truth_floats = np.reshape(np.asarray(truth_floats), [-1])
	pred_floats = np.reshape(np.asarray(pred_floats), [-1])

	precision, recall, _ = precision_recall_curve(truth_floats, pred_vec)

	precision = np.reshape(np.asarray(precision), [-1])
	recall = np.reshape(np.asarray(recall), [-1])

	for number in range(len(recall)):

		if np.sort(recall)[number] <= .95:

			rec95 = np.sort(recall)[number]


	# print('xxxxxxxxxxxxxxxxxx')
	# print('xxxxxxxxxxxxxxxxxx')
	# print('xxxxxxxxxxxxxxxxxx')
	# print('xxxxxxxxxxxxxxxxxx')
	# print('recall')
	# print(rec95)
	# print(recall.shape)
	# print(recall)
	# print('xxxxxxxxxxxxxxxxxx')
	# print('xxxxxxxxxxxxxxxxxx')
	# print('xxxxxxxxxxxxxxxxxx')


	precision_index = np.where(recall == rec95)
	# np.asarray(np.where(np.reshape(recall, [-1])) == rec95)

	# print('xxxxxxxxxxxxxxxxxx')
	# print('xxxxxxxxxxxxxxxxxx')
	# print(rec95)
	# print()
	# print(precision_index)
	# print('xxxxxxxxxxxxxxxxxx')
	# print('xxxxxxxxxxxxxxxxxx')

	# print('xxxxxxxxxxxxxxxxxx')
	# print('xxxxxxxxxxxxxxxxxx')
	# print('xxxxxxxxxxxxxxxxxx')
	# print('xxxxxxxxxxxxxxxxxx')
	# print('precision_index.shape')
	# print(precision_index.shape)
	# print('precision.shape')
	# print(precision.shape)
	# print('precision_index[0]')
	# print(precision_index[0])
	# print('precision_index')
	# print(precision_index)
	# print('precision[precision_index]')
	# print(precision[precision_index])
	# print('xxxxxxxxxxxxxxxxxx')
	# print('xxxxxxxxxxxxxxxxxx')
	# print('xxxxxxxxxxxxxxxxxx')




	return np.mean(precision[precision_index])

def compute_auc(pred_vec, truth_vec):

	truth_vec = np.reshape(np.asarray(truth_vec), [-1])

	truth_ints = []
	
	# truth_ints.append(np.int(truth_vec[number]))

	for number in range(len(truth_vec)):
		if np.float(truth_vec[number]) <= .5:
			truth_ints.append(0)
		else:
			truth_ints.append(1)

	truth_ints = np.reshape(np.asarray(truth_ints), [-1])

	fp_rate, tp_rate, _ = roc_curve(truth_ints, pred_vec, pos_label=1)
	area = auc(fp_rate, tp_rate)

	return area

def compute_roc(pred_vec, truth_vec, num_thresh=2000):

	pred_vec = np.reshape(pred_vec, [-1])
	pred_vec -= np.argmin(pred_vec)
	pred_vec /= (np.argmax(pred_vec) + 1e-6)
	pred_vec *= 1e+5

	# print()
	# print('------------->>> pred_vec', pred_vec.shape)
	# print(pred_vec)
	# print()
	# print()


	truth_vec = np.reshape(truth_vec, [-1])
	threshs = np.linspace(np.argmin(pred_vec), np.argmax(pred_vec), num_thresh)

	total_positives = np.sum(truth_vec)
	total_negatives = len(truth_vec) - total_positives

	acc_vec = np.zeros((len(threshs), 2))
	roc_vec = np.zeros((len(threshs), 2))
	pre_vec = np.zeros((len(threshs), 2))

	for ii in range(len(threshs)):

		bin_preds = np.ones(len(pred_vec))
		bin_preds[np.where(pred_vec <= threshs[ii])] = 0.

		true_positives = np.sum(truth_vec * bin_preds)
		false_positives = np.sum((1-truth_vec) * bin_preds)
		false_negatives = np.sum(truth_vec * (1-bin_preds))
		true_negatives = np.sum((1-truth_vec) * (1-bin_preds))

		tp_rate = true_positives / total_positives
		fp_rate = false_positives / total_negatives

		roc_vec[ii,:] = [fp_rate, tp_rate]
		acc_vec[ii,:] = [tp_rate, (true_positives + true_negatives)/(total_positives + total_negatives)]
		pre_vec[ii,:] = [tp_rate, true_positives/(true_positives + false_negatives)]

	return roc_vec, acc_vec, pre_vec

def roc_Hb1Ac(hb1ac_cont_preds, hb1ac_class_labels):

	hb1ac_class_labels = np.reshape(np.asarray(hb1ac_class_labels), [-1])
	hb1ac_cont_preds = np.reshape(np.asarray(hb1ac_cont_preds), [-1])

	class_labels = np.zeros((len(hb1ac_cont_preds)))
	for i in range(class_labels.shape[0]):
		class_labels[i] = float(hb1ac_class_labels[i])
	class_labels = np.reshape(class_labels, [-1])

	hb1ac_total_positives = np.sum(class_labels)
	hb1ac_total_negatives = len(class_labels) - hb1ac_total_positives

	thresh_array = np.reshape(np.asarray(np.sort(np.unique(hb1ac_cont_preds))), [-1])

	accuracy_vec = np.zeros((len(thresh_array)))

	stats_hb1ac = np.zeros((len(thresh_array), 4))
	roc_vals = np.zeros((len(thresh_array), 2))

	preds_thresh = hb1ac_cont_preds

	for i in range(len(thresh_array)):

		thresh = thresh_array[i]


		preds_thresh2 = np.ones(len(hb1ac_cont_preds))


		preds_thresh2[np.where(preds_thresh <= thresh)] = 0.


		tp_hb1ac = np.sum(class_labels * preds_thresh2)
		fp_hb1ac = np.sum((1-class_labels) * preds_thresh2)
		fn_hb1ac = np.sum(class_labels * (1-preds_thresh2))
		tn_hb1ac = np.sum((1-class_labels) * (1-preds_thresh2))


		stats_hb1ac[i,:] = [tp_hb1ac, fp_hb1ac, fn_hb1ac, tn_hb1ac]

		roc_vals[i,:] = [tp_hb1ac/hb1ac_total_positives, fp_hb1ac/hb1ac_total_negatives]

		accuracy_vec[i] = (tp_hb1ac + tn_hb1ac) / (tp_hb1ac + fp_hb1ac + fn_hb1ac + tn_hb1ac)


	return [np.reshape(roc_vals[:,1], [-1]), np.reshape(roc_vals[:,0], [-1])], accuracy_vec

# def roc_Hb1Ac(hb1ac_cont_preds, hb1ac_class_labels):

	# hb1ac_class_labels = np.reshape(hb1ac_class_labels, [-1])
	# class_labels = np.zeros((len(hb1ac_cont_preds)))
	# for i in range(class_labels.shape[0]):
	# 	class_labels[i] = float(hb1ac_class_labels[i])
	# hb1ac_total_positives = np.sum(class_labels)
	# hb1ac_total_negatives = len(class_labels) - hb1ac_total_positives


	# thresh_array = np.asarray(np.sort(np.unique(hb1ac_cont_preds)))

	# stats_hb1ac = np.zeros((len(thresh_array), 4))
	# roc_vals = np.zeros((len(thresh_array), 2))
	# accuracy_vec = np.zeros((len(thresh_array)))

	# for i in range(len(thresh_array)):

	# 	thresh = thresh_array[i]
	# 	# print('threshold: ', thresh)

	# 	preds_thresh = np.reshape(np.asarray(hb1ac_cont_preds), [-1])

	# 	preds_thresh2 = np.ones(len(hb1ac_cont_preds))

	# 	preds_thresh2[np.where(preds_thresh <= thresh)] = 0.

	# 	tp_hb1ac = np.sum(class_labels * preds_thresh2)
	# 	fp_hb1ac = np.sum((1-class_labels) * preds_thresh2)
	# 	fn_hb1ac = np.sum(class_labels * (1-preds_thresh2))
	# 	tn_hb1ac = np.sum((1-class_labels) * (1-preds_thresh2))

	# 	stats_hb1ac[i,:] = [tp_hb1ac, fp_hb1ac, fn_hb1ac, tn_hb1ac]

	# 	roc_vals[i,:] = [tp_hb1ac/hb1ac_total_positives, fp_hb1ac/hb1ac_total_negatives]

	# 	accuracy_vec[i] = (tp_hb1ac + tn_hb1ac) / (tp_hb1ac + fp_hb1ac + fn_hb1ac + tn_hb1ac)

	# return [np.reshape(roc_vals[:,1], [-1]), np.reshape(roc_vals[:,0], [-1])], accuracy_vec

def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv3d(inputs=input, 
        						   filters=filter, 
        						   kernel_size=kernel, 
        						   strides=stride, 
        						   padding='SAME',
        						   kernel_initializer=kernel_init,
        						   bias_initializer=tf.zeros_initializer(),
        						   name=layer_name)

        with tf.variable_scope(layer_name, reuse=True):
        	# conv_kernel = tf.get_variable('kernel')
        	tf.summary.histogram(layer_name, tf.get_variable('kernel'))

        return network

def Global_Average_Pooling(x, stride=1):
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    depth = np.shape(x)[3]
    pool_size = [width, height, depth]
    return tf.layers.average_pooling3d(inputs=x, pool_size=pool_size, strides=stride) # The stride value does not matter

def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.99,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))

def Drop_out(x, rate, training) :

    return tf.layers.dropout(inputs=x, rate=rate, training=training)

def elu(x):

    return tf.nn.elu(x)

def Average_pooling(x, pool_size=[2,2,2], stride=2, padding='SAME'):

    return tf.layers.average_pooling3d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Max_Pooling(x, pool_size=[3,3,3], stride=2, padding='SAME'):

    return tf.layers.max_pooling3d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Concatenation(layers) :

    return tf.concat(layers, axis=4)

def Linear_all_inone(x) :

	with tf.name_scope('DenseLayers'):

		if second_dense_layer:

			x = tf.layers.dense(
				inputs=x, 
				units=2*dense_units, 
				name='Linear_all_cont0',
				activation=tf.nn.elu,
				kernel_initializer=kernel_init_lin,
				bias_initializer=tf.zeros_initializer())

			with tf.variable_scope('Linear_all_cont0', reuse=True):
				kernel_all_cont0 = tf.get_variable('kernel')
			tf.summary.histogram('kernel_all_cont0', kernel_all_cont0)

			x = tf.layers.dense(
				inputs=x, 
				units=dense_units, 
				name='Linear_all_cont1',
				activation=tf.nn.elu,
				kernel_initializer=kernel_init_lin,
				bias_initializer=tf.zeros_initializer())

			with tf.variable_scope('Linear_all_cont1', reuse=True):
				kernel_all_cont1 = tf.get_variable('kernel')
			tf.summary.histogram('kernel_all_cont1', kernel_all_cont1)

		x_emb = x
		tf.summary.histogram('x_embedding', x_emb)

		x = tf.layers.dense(
			inputs=x, 
			units=10, 
			name='Linear_all_cont',
			# activation=tf.nn.sigmoid,
			kernel_initializer=kernel_init_lin,
			bias_initializer=tf.zeros_initializer())

		with tf.variable_scope('Linear_all_cont', reuse=True):
			kernel_all_cont = tf.get_variable('kernel')
		tf.summary.histogram('kernel_all_cont', kernel_all_cont)

	return x, x_emb

def Linear_all2(x) :

	#################################################################################################
	########################### DenseLayers: ConvLayers --> EmbeddingLayer ##########################
	#################################################################################################


	with tf.name_scope('DenseLayers'):

		if second_dense_layer:

			x = tf.layers.dense(
				inputs=x, 
				units=4*dense_units, 
				name='Linear_all_cont0',
				activation=tf.nn.elu,
				kernel_initializer=kernel_init_lin,
				bias_initializer=tf.zeros_initializer())

			with tf.variable_scope('Linear_all_cont0', reuse=True):
				kernel_all_cont0 = tf.get_variable('kernel')
			tf.summary.histogram('kernel_all_cont0', kernel_all_cont0)

			x = tf.layers.dense(
				inputs=x, 
				units=2*dense_units, 
				name='Linear_all_cont10',
				activation=tf.nn.elu,
				kernel_initializer=kernel_init_lin,
				bias_initializer=tf.zeros_initializer())

			x = tf.layers.dense(
				inputs=x, 
				units=2*dense_units, 
				name='Linear_all_cont1',
				activation=tf.nn.elu,
				kernel_initializer=kernel_init_lin,
				bias_initializer=tf.zeros_initializer())

			with tf.variable_scope('Linear_all_cont1', reuse=True):
				kernel_all_cont1 = tf.get_variable('kernel')
			tf.summary.histogram('kernel_all_cont1', kernel_all_cont1)

			x = tf.layers.dense(
				inputs=x, 
				units=dense_units, 
				name='Linear_all_cont2',
				activation=tf.nn.elu,
				kernel_initializer=kernel_init_lin,
				bias_initializer=tf.zeros_initializer())

			with tf.variable_scope('Linear_all_cont2', reuse=True):
				kernel_all_cont2 = tf.get_variable('kernel')
			tf.summary.histogram('kernel_all_cont2', kernel_all_cont2)

		x_emb = x
		tf.summary.histogram('x_embedding', x_emb)



	#################################################################################################
	########################## DenseLayers: EmbeddingLayer --> OutputNodes ##########################
	#################################################################################################


	with tf.name_scope('final_nodes'):


		#################################################################################################
		############################################## age ##############################################
		#################################################################################################

		x_age = tf.layers.dense(
			inputs=x_emb, 
			units=1, 
			name='Linear_age',
			activation=tf.nn.sigmoid,
			kernel_initializer=kernel_init_lin,
			bias_initializer=tf.zeros_initializer())

		with tf.variable_scope('Linear_age', reuse=True):
			kernel_age = tf.get_variable('kernel')
		tf.summary.histogram('kernel_age', kernel_age)


		#################################################################################################
		############################################## bmi ##############################################
		#################################################################################################

		x_bmi = tf.layers.dense(
			inputs=x_emb, 
			units=1, 
			name='Linear_bmi',
			activation=tf.nn.sigmoid,
			kernel_initializer=kernel_init_lin,
			bias_initializer=tf.zeros_initializer())

		with tf.variable_scope('Linear_bmi', reuse=True):
			kernel_bmi = tf.get_variable('kernel')
		tf.summary.histogram('kernel_bmi', kernel_bmi)

		#################################################################################################
		############################################ fat_imp ############################################
		#################################################################################################

		x_fatimp = tf.layers.dense(
			inputs=x_emb, 
			units=1, 
			name='Linear_fatimp',
			activation=tf.nn.sigmoid,
			kernel_initializer=kernel_init_lin,
			bias_initializer=tf.zeros_initializer())

		with tf.variable_scope('Linear_fatimp', reuse=True):
			kernel_fatimp = tf.get_variable('kernel')
		tf.summary.histogram('kernel_fatimp', kernel_fatimp)

		#################################################################################################
		############################################## sex ##############################################
		#################################################################################################

		x_sex = tf.layers.dense(
			inputs=x_emb, 
			units=2, 
			name='Linear_sex',
			activation=tf.nn.sigmoid,
			kernel_initializer=kernel_init_lin,
			bias_initializer=tf.zeros_initializer())

		with tf.variable_scope('Linear_sex', reuse=True):
			kernel_sex = tf.get_variable('kernel')
		tf.summary.histogram('kernel_sex', kernel_sex)

		#################################################################################################
		############################################ insulin ############################################
		#################################################################################################


		ins_layer_input = tf.concat([
			x_age,
			x_bmi,
			x_fatimp,
			x_sex], axis=1)

		ins_layer_input = tf.layers.dense(
			inputs=ins_layer_input, 
			units=dense_units, 
			name='ins_layer_input',
			activation=tf.nn.elu,
			kernel_initializer=kernel_init_lin,
			bias_initializer=tf.zeros_initializer())

		ins_layer_input = tf.concat([
			x_emb,
			ins_layer_input], axis=1)


		ins_layer_input = tf.layers.dense(
			inputs=ins_layer_input, 
			units=dense_units, 
			name='ins_layer_input1',
			activation=tf.nn.elu,
			kernel_initializer=kernel_init_lin,
			bias_initializer=tf.zeros_initializer())


		x_insulin = tf.layers.dense(
			inputs=ins_layer_input, 
			units=1, 
			name='Linear_insulin',
			activation=tf.nn.sigmoid,
			kernel_initializer=kernel_init_lin,
			bias_initializer=tf.zeros_initializer())

		with tf.variable_scope('Linear_insulin', reuse=True):
			kernel_insulin = tf.get_variable('kernel')
		tf.summary.histogram('kernel_insulin', kernel_insulin)

		#################################################################################################
		########################################### di_index ############################################
		#################################################################################################

		x_di = tf.layers.dense(
			inputs=ins_layer_input, 
			units=1, 
			name='Linear_di',
			activation=tf.nn.sigmoid,
			kernel_initializer=kernel_init_lin,
			bias_initializer=tf.zeros_initializer())

		with tf.variable_scope('Linear_di', reuse=True):
			kernel_di = tf.get_variable('kernel')
		tf.summary.histogram('kernel_di', kernel_di)

		#################################################################################################
		############################################# hba1c #############################################
		#################################################################################################

		x_hba1c = tf.layers.dense(
			inputs=ins_layer_input, 
			units=1, 
			name='Linear_hba1c',
			activation=tf.nn.sigmoid,
			kernel_initializer=kernel_init_lin,
			bias_initializer=tf.zeros_initializer())

		with tf.variable_scope('Linear_hba1c', reuse=True):
			kernel_hba1c = tf.get_variable('kernel')
		tf.summary.histogram('kernel_hba1c', kernel_hba1c)

		#################################################################################################
		########################################### diabetes ############################################
		#################################################################################################

		diab_layer_input = tf.concat([
			x_age,
			x_bmi,
			x_fatimp,
			x_insulin,
			x_di,
			x_hba1c,
			x_sex], axis=1)

		diab_layer_input = tf.layers.dense(
			inputs=diab_layer_input, 
			units=dense_units, 
			name='diab_layer_input',
			activation=tf.nn.elu,
			kernel_initializer=kernel_init_lin,
			bias_initializer=tf.zeros_initializer())

		diab_layer_input = tf.concat([
			x_emb,
			diab_layer_input], axis=1)

		x_diab = tf.layers.dense(
			inputs=diab_layer_input, 
			units=dense_units, 
			name='Linear_diab0',
			activation=tf.nn.elu,
			kernel_initializer=kernel_init_lin,
			bias_initializer=tf.zeros_initializer())

		with tf.variable_scope('Linear_diab0', reuse=True):
			kernel_diab0 = tf.get_variable('kernel')
		tf.summary.histogram('kernel_diab0', kernel_diab0)

		x_diab = tf.layers.dense(
			inputs=diab_layer_input, 
			units=2, 
			name='Linear_diab',
			activation=tf.nn.sigmoid,
			kernel_initializer=kernel_init_lin,
			bias_initializer=tf.zeros_initializer())

		with tf.variable_scope('Linear_diab', reuse=True):
			kernel_diab = tf.get_variable('kernel')
		tf.summary.histogram('kernel_diab', kernel_diab)



		#################################################################################################
		###################################### concatenated output ######################################
		#################################################################################################


		x = tf.concat([
			x_age,
			x_bmi,
			x_fatimp,
			x_insulin,
			x_di,
			x_hba1c,
			x_diab,
			x_sex], axis=1)


	return x, x_emb

def Linear_all(x) :

	#################################################################################################
	########################### DenseLayers: ConvLayers --> EmbeddingLayer ##########################
	#################################################################################################


	with tf.name_scope('DenseLayers'):

		if second_dense_layer:

			x = tf.layers.dense(
				inputs=x, 
				units=4*dense_units, 
				name='Linear_all_cont0',
				activation=tf.nn.elu,
				kernel_initializer=kernel_init_lin,
				bias_initializer=tf.zeros_initializer())

			with tf.variable_scope('Linear_all_cont0', reuse=True):
				kernel_all_cont0 = tf.get_variable('kernel')
			tf.summary.histogram('kernel_all_cont0', kernel_all_cont0)

			x = tf.layers.dense(
				inputs=x, 
				units=2*dense_units, 
				name='Linear_all_cont10',
				activation=tf.nn.elu,
				kernel_initializer=kernel_init_lin,
				bias_initializer=tf.zeros_initializer())

			x = tf.layers.dense(
				inputs=x, 
				units=2*dense_units, 
				name='Linear_all_cont1',
				activation=tf.nn.elu,
				kernel_initializer=kernel_init_lin,
				bias_initializer=tf.zeros_initializer())

			with tf.variable_scope('Linear_all_cont1', reuse=True):
				kernel_all_cont1 = tf.get_variable('kernel')
			tf.summary.histogram('kernel_all_cont1', kernel_all_cont1)

			x = tf.layers.dense(
				inputs=x, 
				units=dense_units, 
				name='Linear_all_cont2',
				activation=tf.nn.elu,
				kernel_initializer=kernel_init_lin,
				bias_initializer=tf.zeros_initializer())

			with tf.variable_scope('Linear_all_cont2', reuse=True):
				kernel_all_cont2 = tf.get_variable('kernel')
			tf.summary.histogram('kernel_all_cont2', kernel_all_cont2)

		x_emb = x
		tf.summary.histogram('x_embedding', x_emb)



	#################################################################################################
	########################## DenseLayers: EmbeddingLayer --> OutputNodes ##########################
	#################################################################################################


	with tf.name_scope('final_nodes'):


		#################################################################################################
		############################################## age ##############################################
		#################################################################################################

		x_age = tf.layers.dense(
			inputs=x_emb, 
			units=1, 
			name='Linear_age',
			activation=tf.nn.sigmoid,
			kernel_initializer=kernel_init_lin,
			bias_initializer=tf.zeros_initializer())

		with tf.variable_scope('Linear_age', reuse=True):
			kernel_age = tf.get_variable('kernel')
		tf.summary.histogram('kernel_age', kernel_age)


		#################################################################################################
		############################################## bmi ##############################################
		#################################################################################################

		x_bmi = tf.layers.dense(
			inputs=x_emb, 
			units=1, 
			name='Linear_bmi',
			activation=tf.nn.sigmoid,
			kernel_initializer=kernel_init_lin,
			bias_initializer=tf.zeros_initializer())

		with tf.variable_scope('Linear_bmi', reuse=True):
			kernel_bmi = tf.get_variable('kernel')
		tf.summary.histogram('kernel_bmi', kernel_bmi)


		#################################################################################################
		############################################ fat_imp ############################################
		#################################################################################################

		x_fatimp = tf.layers.dense(
			inputs=x_emb, 
			units=1, 
			name='Linear_fatimp',
			activation=tf.nn.sigmoid,
			kernel_initializer=kernel_init_lin,
			bias_initializer=tf.zeros_initializer())

		with tf.variable_scope('Linear_fatimp', reuse=True):
			kernel_fatimp = tf.get_variable('kernel')
		tf.summary.histogram('kernel_fatimp', kernel_fatimp)


		#################################################################################################
		############################################ insulin ############################################
		#################################################################################################

		x_insulin = tf.layers.dense(
			inputs=x_emb, 
			units=1, 
			name='Linear_insulin',
			activation=tf.nn.sigmoid,
			kernel_initializer=kernel_init_lin,
			bias_initializer=tf.zeros_initializer())

		with tf.variable_scope('Linear_insulin', reuse=True):
			kernel_insulin = tf.get_variable('kernel')
		tf.summary.histogram('kernel_insulin', kernel_insulin)


		#################################################################################################
		########################################### di_index ############################################
		#################################################################################################

		x_di = tf.layers.dense(
			inputs=x_emb, 
			units=1, 
			name='Linear_di',
			activation=tf.nn.sigmoid,
			kernel_initializer=kernel_init_lin,
			bias_initializer=tf.zeros_initializer())

		with tf.variable_scope('Linear_di', reuse=True):
			kernel_di = tf.get_variable('kernel')
		tf.summary.histogram('kernel_di', kernel_di)


		#################################################################################################
		############################################# hba1c #############################################
		#################################################################################################

		x_hba1c = tf.layers.dense(
			inputs=x_emb, 
			units=1, 
			name='Linear_hba1c',
			activation=tf.nn.sigmoid,
			kernel_initializer=kernel_init_lin,
			bias_initializer=tf.zeros_initializer())

		with tf.variable_scope('Linear_hba1c', reuse=True):
			kernel_hba1c = tf.get_variable('kernel')
		tf.summary.histogram('kernel_hba1c', kernel_hba1c)


		#################################################################################################
		############################################## sex ##############################################
		#################################################################################################

		x_sex = tf.layers.dense(
			inputs=x_emb, 
			units=2, 
			name='Linear_sex',
			activation=tf.nn.sigmoid,
			kernel_initializer=kernel_init_lin,
			bias_initializer=tf.zeros_initializer())

		with tf.variable_scope('Linear_sex', reuse=True):
			kernel_sex = tf.get_variable('kernel')
		tf.summary.histogram('kernel_sex', kernel_sex)



		#################################################################################################
		########################################### diabetes ############################################
		#################################################################################################

		diab_layer_input = tf.concat([
			x_age,
			x_bmi,
			x_fatimp,
			x_insulin,
			x_di,
			x_hba1c,
			x_sex], axis=1)

		diab_layer_input = tf.layers.dense(
			inputs=diab_layer_input, 
			units=dense_units, 
			name='diab_layer_input',
			activation=tf.nn.elu,
			kernel_initializer=kernel_init_lin,
			bias_initializer=tf.zeros_initializer())

		diab_layer_input = tf.concat([
			x_emb,
			diab_layer_input], axis=1)


		# x_diab = tf.layers.dense(
		# 	inputs=diab_layer_input, 
		# 	units=dense_units, 
		# 	name='Linear_diab0',
		# 	activation=tf.nn.elu,
		# 	kernel_initializer=kernel_init_lin,
		# 	bias_initializer=tf.zeros_initializer())

		# with tf.variable_scope('Linear_diab0', reuse=True):
		# 	kernel_diab0 = tf.get_variable('kernel')
		# tf.summary.histogram('kernel_diab0', kernel_diab0)

		x_diab = tf.layers.dense(
			inputs=diab_layer_input, 
			units=2, 
			name='Linear_diab',
			activation=tf.nn.sigmoid,
			kernel_initializer=kernel_init_lin,
			bias_initializer=tf.zeros_initializer())

		with tf.variable_scope('Linear_diab', reuse=True):
			kernel_diab = tf.get_variable('kernel')
		tf.summary.histogram('kernel_diab', kernel_diab)



		#################################################################################################
		###################################### concatenated output ######################################
		#################################################################################################


		x = tf.concat([
			x_age,
			x_bmi,
			x_fatimp,
			x_insulin,
			x_di,
			x_hba1c,
			x_diab,
			x_sex], axis=1)


	return x, x_emb

def Linear_insulin(x) :

	with tf.name_scope('DenseLayers'):

		if second_dense_layer:

			x = tf.layers.dense(
				inputs=x, 
				units=dense_units, 
				name='Linear_insulin0',
				activation=tf.nn.elu,
				kernel_initializer=kernel_init_lin,
				bias_initializer=tf.zeros_initializer())

			with tf.variable_scope('Linear_insulin0', reuse=True):
				kernel_insulin0 = tf.get_variable('kernel')
			tf.summary.histogram('kernel_insulin0', kernel_insulin0)

			# x = tf.layers.dense(
			# 	inputs=x, 
			# 	units=2*dense_units, 
			# 	name='Linear_insulin1',
			# 	activation=tf.nn.elu,
			# 	kernel_initializer=kernel_init_lin,
			# 	bias_initializer=tf.zeros_initializer())

			# with tf.variable_scope('Linear_insulin1', reuse=True):
			# 	kernel_insulin1 = tf.get_variable('kernel')
			# tf.summary.histogram('kernel_insulin1', kernel_insulin1)

		x_emb = x

		x = tf.layers.dense(
			inputs=x, 
			units=1, 
			name='Linear_insulin',
			activation=tf.nn.sigmoid,
			kernel_initializer=kernel_init_lin,
			bias_initializer=tf.zeros_initializer())

		with tf.variable_scope('Linear_insulin', reuse=True):
			kernel_insulin = tf.get_variable('kernel')
		tf.summary.histogram('kernel_insulin', kernel_insulin)

	return x, x_emb

def Linear_DI(x) :

	with tf.name_scope('DenseLayers'):

		if second_dense_layer:

			x = tf.layers.dense(
				inputs=x, 
				units=dense_units, 
				name='Linear_DI0',
				activation=tf.nn.elu,
				kernel_initializer=kernel_init_lin,
				bias_initializer=tf.zeros_initializer())

			with tf.variable_scope('Linear_DI0', reuse=True):
				kernel_DI0 = tf.get_variable('kernel')
			tf.summary.histogram('kernel_DI0', kernel_DI0)

			# x = tf.layers.dense(
			# 	inputs=x, 
			# 	units=2*dense_units, 
			# 	name='Linear_DI1',
			# 	activation=tf.nn.elu,
			# 	kernel_initializer=kernel_init_lin,
			# 	bias_initializer=tf.zeros_initializer())

			# with tf.variable_scope('Linear_DI1', reuse=True):
			# 	kernel_DI1 = tf.get_variable('kernel')
			# tf.summary.histogram('kernel_DI1', kernel_DI1)

		x_emb = x

		x = tf.layers.dense(
			inputs=x, 
			units=1, 
			name='Linear_DI',
			activation=tf.nn.sigmoid,
			kernel_initializer=kernel_init_lin,
			bias_initializer=tf.zeros_initializer())

		with tf.variable_scope('Linear_DI', reuse=True):
			kernel_DI = tf.get_variable('kernel')
		tf.summary.histogram('kernel_DI', kernel_DI)

	return x, x_emb

def Linear_diab_cont(x) :

	with tf.name_scope('DenseLayers'):

		if second_dense_layer:

			x = tf.layers.dense(
				inputs=x, 
				units=dense_units, 
				name='Linear_diab_cont0',
				activation=tf.nn.elu,
				kernel_initializer=kernel_init_lin,
				bias_initializer=tf.zeros_initializer())

			with tf.variable_scope('Linear_diab_cont0', reuse=True):
				kernel_diab_cont = tf.get_variable('kernel')
			tf.summary.histogram('kernel_diab_cont', kernel_diab_cont)

			# x = tf.layers.dense(
			# 	inputs=x, 
			# 	units=2*dense_units, 
			# 	name='Linear_diab_cont1',
			# 	activation=tf.nn.elu,
			# 	kernel_initializer=kernel_init_lin,
			# 	bias_initializer=tf.zeros_initializer())

			# with tf.variable_scope('Linear_diab_cont1', reuse=True):
			# 	kernel_diab_cont1 = tf.get_variable('kernel')
			# tf.summary.histogram('kernel_diab_cont1', kernel_diab_cont1)



		x_emb = x


		x = tf.layers.dense(
			inputs=x, 
			units=1, 
			name='Linear_diab_cont2',
			activation=tf.nn.sigmoid,
			kernel_initializer=kernel_init_lin,
			bias_initializer=tf.zeros_initializer())

		with tf.variable_scope('Linear_diab_cont2', reuse=True):
			kernel_diab_cont = tf.get_variable('kernel')
		tf.summary.histogram('kernel_diab_cont', kernel_diab_cont)

	return x, x_emb

def Linear_diab_all(x) :

	with tf.name_scope('DenseLayers'):

		if second_dense_layer:

			x = tf.layers.dense(
				inputs=x, 
				units=dense_units, 
				name='Linear_diab_all0',
				activation=tf.nn.elu,
				kernel_initializer=kernel_init_lin,
				bias_initializer=tf.zeros_initializer())

			with tf.variable_scope('Linear_diab_all0', reuse=True):
				kernel_diab_all0 = tf.get_variable('kernel')
			tf.summary.histogram('kernel_diab_all0', kernel_diab_all0)


			# x = tf.layers.dense(
			# 	inputs=x, 
			# 	units=dense_units, 
			# 	name='Linear_diab_all1',
			# 	activation=tf.nn.elu,
			# 	kernel_initializer=kernel_init_lin,
			# 	bias_initializer=tf.zeros_initializer())

			# with tf.variable_scope('Linear_diab_all1', reuse=True):
			# 	kernel_diab_all1 = tf.get_variable('kernel')
			# tf.summary.histogram('kernel_diab_all1', kernel_diab_all1)

		x_emb = x

		x = tf.layers.dense(
			inputs=x, 
			units=2, 
			name='Linear_diab_all',
			# activation=tf.nn.elu,
			kernel_initializer=kernel_init_lin,
			bias_initializer=tf.zeros_initializer())

		with tf.variable_scope('Linear_diab_all', reuse=True):
			kernel_diab_all = tf.get_variable('kernel')
		tf.summary.histogram('kernel_diab_all', kernel_diab_all)

	return x, x_emb

class DenseNet():

	def __init__(self, x, add_feats, filters, training):
		self.filters = filters
		self.training = training
		self.model = self.Dense_net(x, add_feats)

	def bottleneck_layer(self, x, scope):

		# with tf.name_scope(scope):
		x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
		x = elu(x)
		x = conv_layer(x, filter=4 * self.filters, kernel=[1,1,1], layer_name=scope+'_conv1')
		x = Drop_out(x, rate=dropout_rate, training=self.training)

		x = Batch_Normalization(x, training=self.training, scope=scope+'_batch2')
		x = elu(x)
		x = conv_layer(x, filter=self.filters, kernel=[3,3,3], layer_name=scope+'_conv2')
		x = Drop_out(x, rate=dropout_rate, training=self.training)

		return x

	def transition_layer(self, x, scope, conv_stride=2):

		# with tf.name_scope(scope):
		x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
		x = elu(x)
		x = conv_layer(x, filter=self.filters, kernel=[1,1,1], layer_name=scope+'_conv1')
		x = Drop_out(x, rate=dropout_rate, training=self.training)
		x = Average_pooling(x, pool_size=[2,2,2], stride=conv_stride)

		return x

	def dense_block(self, input_x, nb_layers, layer_name):

		with tf.name_scope(layer_name):

			layers_concat = list()
			layers_concat.append(input_x)

			x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))

			layers_concat.append(x)

			for i in range(nb_layers - 1):
				x = Concatenation(layers_concat)
				x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
				layers_concat.append(x)

			x = Concatenation(layers_concat)

			return x

	def Dense_net(self, input_x, add_feats):


		with tf.name_scope('inital_conv_layer'):

			if outprint:
				for space in range(4): print()
				print('--------------------------------------------------------------------')
				print('-------------------- dimensions through network --------------------')
				print('--------------------------------------------------------------------')
				printer(input_x, 'input:')

			x = Max_Pooling(input_x, pool_size=[2,2,2], stride=1)
			if outprint: printer(x, 'after first max_pooling')

			x = Batch_Normalization(x, training=self.training, scope='BN0')
			if outprint: printer(x, 'after first Batch_Normalization')

			x = conv_layer(x, filter=first_conv_filters, kernel=[5,5,5], stride=2, layer_name='conv0')
			if outprint: printer(x, 'after first conv_layer')

			# x = Max_Pooling(x, pool_size=[2,2,2], stride=first_maxpool_stride)
			x = elu(x)
			if outprint: printer(x, 'after first elu')

			x_in = x

			# tf.summary.image('filter0',tf.reshape(tf.reduce_mean(x_in[:,:,:,:,0], axis=2), [batch_size, x_in.shape[1], x_in.shape[3], 1]), max_outputs=10)
			# tf.summary.image('filter1',tf.reshape(tf.reduce_mean(x_in[:,:,:,:,1], axis=2), [batch_size, x_in.shape[1], x_in.shape[3], 1]), max_outputs=10)
			# tf.summary.image('filter2',tf.reshape(tf.reduce_mean(x_in[:,:,:,:,2], axis=2), [batch_size, x_in.shape[1], x_in.shape[3], 1]), max_outputs=10)
			# tf.summary.image('filter3',tf.reshape(tf.reduce_mean(x_in[:,:,:,:,3], axis=2), [batch_size, x_in.shape[1], x_in.shape[3], 1]), max_outputs=10)


		if num_dense_blocks > 0:

			with tf.name_scope('dense_block1'):
				x = self.dense_block(input_x=x, nb_layers=num_layers_1st_dense, layer_name='dense_1')
				if outprint: printer(x, 'after dense_block1')

			with tf.name_scope('transition1'):
				x = self.transition_layer(x, scope='trans_1')
				if outprint: printer(x, 'after transition1')



			if num_dense_blocks > 1:

				with tf.name_scope('dense_block2'):
					x = self.dense_block(input_x=x, nb_layers=num_layers_2nd_dense, layer_name='dense_2')
					if outprint: printer(x, 'after dense_block2')

				with tf.name_scope('transition2'):
					x = self.transition_layer(x, scope='trans_2')
					if outprint: printer(x, 'after transition2')



				if num_dense_blocks > 2:

					with tf.name_scope('dense_block3'):
						x = self.dense_block(input_x=x, nb_layers=num_layers_3rd_dense, layer_name='dense_3')
						if outprint: printer(x, 'after dense_block3')

					with tf.name_scope('transition3'):
						x = self.transition_layer(x, scope='trans_3')
						if outprint: printer(x, 'after transition3')
							


					if num_dense_blocks > 3:

						with tf.name_scope('dense_block4'):
							x = self.dense_block(input_x=x, nb_layers=num_layers_4th_dense, layer_name='dense_4')
							if outprint: printer(x, 'after dense_block4')

						with tf.name_scope('transition4'):
							x = self.transition_layer(x, scope='trans_4')
							if outprint: printer(x, 'after transition4')



		with tf.name_scope('final_layer'):

			# x = Max_Pooling(x, pool_size=[2,2,2], stride=2)
			# if outprint:
				# print()
				# print('after final max_pooling')
				# print(x)
				# print()

			x = Batch_Normalization(x, training=self.training, scope='final_batchNorm')
			if outprint: printer(x, 'after final Batch_Normalization')

			x = elu(x)
			if outprint: printer(x, 'after final elu')
				
			x = flatten(x)
			if outprint: printer(x, 'after flattening')


		if additional_features:

			with tf.name_scope('add_features'):

				add_feats = tf.layers.dense(
					inputs=add_feats, 
					units=dense_units, 
					name='Linear_add_feats',
					activation=tf.nn.elu,
					kernel_initializer=kernel_init_lin,
					bias_initializer=tf.zeros_initializer())

				with tf.variable_scope('Linear_add_feats', reuse=True):
					kernel_add_feats = tf.get_variable('kernel')
				tf.summary.histogram('kernel_add_feats', kernel_add_feats)

				add_feats = Drop_out(add_feats, rate=dropout_rate, training=self.training)

				add_feats = tf.layers.dense(
					inputs=add_feats, 
					units=2*dense_units, 
					name='Linear_add_feats1',
					# activation=tf.nn.elu,
					kernel_initializer=kernel_init_lin,
					bias_initializer=tf.zeros_initializer())

				with tf.variable_scope('Linear_add_feats1', reuse=True):
					kernel_add_feats1 = tf.get_variable('kernel')
				tf.summary.histogram('kernel_add_feats1', kernel_add_feats1)

			with tf.name_scope('concat'):

				x = tf.concat([x, add_feats], axis=1)


		

		x_out, x_emb = Linear_all(x)



		out_gradients_age = tf.gradients(x_out[:,0], [input_x])
		grad_times_in_age = out_gradients_age * input_x
		tf.summary.image('grad_times_in_age',tf.reshape(tf.reduce_mean(grad_times_in_age[0,:,:,:,:,0], axis=2), [batch_size, grad_times_in_age.shape[2], grad_times_in_age.shape[4], 1]), max_outputs=30)

		out_gradients_bmi = tf.gradients(x_out[:,1], [input_x])
		grad_times_in_bmi = out_gradients_bmi * input_x
		tf.summary.image('grad_times_in_bmi',tf.reshape(tf.reduce_mean(grad_times_in_bmi[0,:,:,:,:,0], axis=2), [batch_size, grad_times_in_bmi.shape[2], grad_times_in_bmi.shape[4], 1]), max_outputs=30)

		out_gradients_fatimp = tf.gradients(x_out[:,2], [input_x])
		grad_times_in_fatimp = out_gradients_fatimp * input_x
		tf.summary.image('grad_times_in_fatimp',tf.reshape(tf.reduce_mean(grad_times_in_fatimp[0,:,:,:,:,0], axis=2), [batch_size, grad_times_in_fatimp.shape[2], grad_times_in_fatimp.shape[4], 1]), max_outputs=30)

		out_gradients_insulin = tf.gradients(x_out[:,3], [input_x])
		grad_times_in_insulin = out_gradients_insulin * input_x
		tf.summary.image('grad_times_in_insulin',tf.reshape(tf.reduce_mean(grad_times_in_insulin[0,:,:,:,:,0], axis=2), [batch_size, grad_times_in_insulin.shape[2], grad_times_in_insulin.shape[4], 1]), max_outputs=30)

		out_gradients_di = tf.gradients(x_out[:,4], [input_x])
		grad_times_in_di = out_gradients_di * input_x
		tf.summary.image('grad_times_in_di',tf.reshape(tf.reduce_mean(grad_times_in_di[0,:,:,:,:,0], axis=2), [batch_size, grad_times_in_di.shape[2], grad_times_in_di.shape[4], 1]), max_outputs=30)

		out_gradients_hba1c = tf.gradients(x_out[:,5], [input_x])
		grad_times_in_hba1c = out_gradients_hba1c * input_x
		tf.summary.image('grad_times_in_hba1c',tf.reshape(tf.reduce_mean(grad_times_in_hba1c[0,:,:,:,:,0], axis=2), [batch_size, grad_times_in_hba1c.shape[2], grad_times_in_hba1c.shape[4], 1]), max_outputs=30)

		out_gradients_diab = tf.gradients(x_out[:,6], [input_x])
		grad_times_in_diab = out_gradients_diab * input_x
		tf.summary.image('grad_times_in_diab',tf.reshape(tf.reduce_mean(grad_times_in_diab[0,:,:,:,:,0], axis=2), [batch_size, grad_times_in_diab.shape[2], grad_times_in_diab.shape[4], 1]), max_outputs=30)

		out_gradients_sex = tf.gradients(x_out[:,8], [input_x])
		grad_times_in_sex = out_gradients_sex * input_x
		tf.summary.image('grad_times_in_sex',tf.reshape(tf.reduce_mean(grad_times_in_sex[0,:,:,:,:,0], axis=2), [batch_size, grad_times_in_sex.shape[2], grad_times_in_sex.shape[4], 1]), max_outputs=30)


		gradients_times_input = tf.concat([
			grad_times_in_age,
			grad_times_in_bmi,
			grad_times_in_fatimp,
			grad_times_in_insulin,
			grad_times_in_di,
			grad_times_in_hba1c,
			grad_times_in_diab,
			grad_times_in_sex
			], 
			axis=0)



		if outprint: 
			printer(x_out, 'x_out')
			printer(gradients_times_input, 'gradients_times_input')
			print('--------------------------------------------------------------------')
			print('--------------------------------------------------------------------')
			print('--------------------------------------------------------------------')
			for space in range(4): print()
			


		return x_out, gradients_times_input, x_emb

class ConvNet():

	def __init__(self, x, add_feats, training):
		self.training = training
		self.model = self.Conv_net(x, add_feats)

	def Conv_net(self, input_x, add_feats_in):

		with tf.name_scope('conv_layer_0'):

			if outprint:
				for space in range(2): print()
				print('------------------------------------------------')
				print('---------- dimensions through network ----------')
				print('------------------------------------------------')
				printer(input_x, 'input:')

			x = Batch_Normalization(input_x, training=self.training, scope='BN0')
			x = elu(x)
			x = conv_layer(x, filter=first_conv_filters, kernel=[3,3,3], stride=1, layer_name='conv0')
			x = Max_Pooling(x, pool_size=[2,2,2], stride=2)

			if outprint: printer(x, 'after 1st layer')


		with tf.name_scope('conv_layer_1'):
							
			x = elu(x)				
			x = conv_layer(x, filter=first_conv_filters, kernel=[3,3,3], stride=1, layer_name='conv1')
			x = Max_Pooling(x, pool_size=[2,2,2], stride=1)

			if outprint: printer(x, 'after 2nd layer')


		with tf.name_scope('conv_layer_2'):
			
			x = elu(x)				
			x = conv_layer(x, filter=2*first_conv_filters, kernel=[3,3,3], stride=1, layer_name='conv2')
			x = Max_Pooling(x, pool_size=[2,2,2], stride=1)
			
			if outprint: printer(x, 'after 3rd layer')

			x_in = x

			tf.summary.image('filter0',tf.reshape(tf.reduce_mean(x_in[:,:,:,:,0], axis=2), [batch_size, x_in.shape[1], x_in.shape[3], 1]), max_outputs=10)
			tf.summary.image('filter1',tf.reshape(tf.reduce_mean(x_in[:,:,:,:,1], axis=2), [batch_size, x_in.shape[1], x_in.shape[3], 1]), max_outputs=10)
			tf.summary.image('filter2',tf.reshape(tf.reduce_mean(x_in[:,:,:,:,2], axis=2), [batch_size, x_in.shape[1], x_in.shape[3], 1]), max_outputs=10)
			tf.summary.image('filter3',tf.reshape(tf.reduce_mean(x_in[:,:,:,:,3], axis=2), [batch_size, x_in.shape[1], x_in.shape[3], 1]), max_outputs=10)


		if num_conv_layers > 3:

			with tf.name_scope('conv_layer_3'):
				
				x = elu(x)				
				x = conv_layer(x, filter=2*first_conv_filters, kernel=[3,3,3], stride=1, layer_name='conv3')
				x = Max_Pooling(x, pool_size=[2,2,2], stride=1)
				
				if outprint: printer(x, 'after 4th layer')


			if num_conv_layers > 4:

				with tf.name_scope('conv_layer_4'):
					
					x = elu(x)				
					x = conv_layer(x, filter=4*first_conv_filters, kernel=[3,3,3], stride=1, layer_name='conv4')
					x = Max_Pooling(x, pool_size=[2,2,2], stride=2)
					
					if outprint: printer(x, 'after 5th layer')


				if num_conv_layers > 5:

					with tf.name_scope('conv_layer_5'):
						
						x = elu(x)				
						x = conv_layer(x, filter=4*first_conv_filters, kernel=[3,3,3], stride=1, layer_name='conv5')
						x = Max_Pooling(x, pool_size=[2,2,2], stride=1)
						
						if outprint: printer(x, 'after 6th layer')


					if num_conv_layers > 6:

						with tf.name_scope('conv_layer_6'):
							
							x = elu(x)				
							x = conv_layer(x, filter=8*first_conv_filters, kernel=[3,3,3], stride=1, layer_name='conv6')
							x = Max_Pooling(x, pool_size=[2,2,2], stride=1)
							
							if outprint: printer(x, 'after 7th layer')


						if num_conv_layers > 7:

							with tf.name_scope('conv_layer_7'):
								
								x = elu(x)				
								x = conv_layer(x, filter=8*first_conv_filters, kernel=[3,3,3], stride=1, layer_name='conv7')
								x = Max_Pooling(x, pool_size=[2,2,2], stride=2)
								
								if outprint: printer(x, 'after 8th layer')


							if num_conv_layers > 8:

								with tf.name_scope('conv_layer_8'):
									
									x = elu(x)				
									x = conv_layer(x, filter=16*first_conv_filters, kernel=[3,3,3], stride=1, layer_name='conv8')
									x = Max_Pooling(x, pool_size=[2,2,2], stride=1)
									
									if outprint: printer(x, 'after 9th layer')


								if num_conv_layers > 9:

									with tf.name_scope('conv_layer_9'):
										
										x = elu(x)				
										x = conv_layer(x, filter=16*first_conv_filters, kernel=[3,3,3], stride=1, layer_name='conv9')
										x = Max_Pooling(x, pool_size=[2,2,2], stride=1)
										
										if outprint: printer(x, 'after 10th layer')


									if num_conv_layers > 10:

										with tf.name_scope('conv_layer_10'):
											
											x = elu(x)				
											x = conv_layer(x, filter=32*first_conv_filters, kernel=[3,3,3], stride=1, layer_name='conv10')
											x = Max_Pooling(x, pool_size=[2,2,2], stride=1)
											
											if outprint: printer(x, 'after 11th layer')


										if num_conv_layers > 11:

											with tf.name_scope('conv_layer_11'):
												
												x = elu(x)				
												x = conv_layer(x, filter=32*first_conv_filters, kernel=[3,3,3], stride=1, layer_name='conv11')
												x = Max_Pooling(x, pool_size=[2,2,2], stride=2)
												
												if outprint: printer(x, 'after 12th layer')





		if additional_features:

			with tf.name_scope('add_features'):

				add_feats = tf.layers.dense(
					inputs=add_feats_in, 
					units=dense_units, 
					name='Linear_add_feats',
					activation=tf.nn.elu,
					kernel_initializer=kernel_init_lin,
					bias_initializer=tf.zeros_initializer())

				with tf.variable_scope('Linear_add_feats', reuse=True):
					kernel_add_feats = tf.get_variable('kernel')
				tf.summary.histogram('kernel_add_feats', kernel_add_feats)

				add_feats = Drop_out(add_feats, rate=dropout_rate, training=self.training)

				add_feats = tf.layers.dense(
					inputs=add_feats, 
					units=2*dense_units, 
					name='Linear_add_feats1',
					activation=tf.nn.elu,
					kernel_initializer=kernel_init_lin,
					bias_initializer=tf.zeros_initializer())

				with tf.variable_scope('Linear_add_feats1', reuse=True):
					kernel_add_feats1 = tf.get_variable('kernel')
				tf.summary.histogram('kernel_add_feats1', kernel_add_feats1)


	


		with tf.name_scope('final_layer'):

			# x = Max_Pooling(x, pool_size=[2,2,2], stride=2)
			# if outprint:
				# print()
				# print('after final max_pooling')
				# print(x)
				# print()

			x = Batch_Normalization(x, training=self.training, scope='final_batchNorm')
			if outprint: printer(x, 'after final Batch_Normalization')

			x = elu(x)
			if outprint: printer(x, 'after final elu')
				
			x = flatten(x)
			if outprint: printer(x, 'after flattening')

			if additional_features:
				with tf.name_scope('combine'):
					x = tf.concat([x, add_feats], axis=1)





		if prediction_label == 'insulin':

			x_out = Linear_insulin(x)

		if prediction_label == 'di_index':

			x_out = Linear_DI(x)

		if prediction_label == 'diab_cont':

			x_out = Linear_diab_cont(x)

		if prediction_label == 'diab_all':

			x_out = Linear_diab_all(x)



		out_gradients_0 = tf.gradients(x_out[:,0], [input_x])
		gradients_times_input = out_gradients_0 * input_x
		
		tf.summary.image('gradients_times_input',tf.reshape(tf.reduce_mean(gradients_times_input[0,:,:,:,:,0], axis=2), [batch_size, gradients_times_input.shape[2], gradients_times_input.shape[4], 1]), max_outputs=20)
		tf.summary.histogram('gradients_times_input_hist', tf.reshape(gradients_times_input, [-1]))



		if outprint: 
			printer(x_out, 'x_out')
			printer(gradients_times_input, 'gradients_times_input')
			print('------------------------------------------------')
			print('------------------------------------------------')
			for space in range(4): print()
			


		return x_out, gradients_times_input

def get_loss_insulin(dense_output, labels):

	with tf.name_scope('insulin_rmse_loss'):
		dense_output_insulin = tf.reshape(dense_output, [batch_size,1])

		labels_insulin = tf.reshape(labels[:,6], [batch_size, 1])

		loss_insulin = tf.losses.mean_squared_error(labels_insulin, dense_output_insulin, scope=None, loss_collection=tf.GraphKeys.LOSSES)#, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS)
		loss_insulin = tf.squeeze(tf.reduce_mean(tf.sqrt(loss_insulin)))


		tf.summary.scalar('loss_insulin', loss_insulin)
		tf.summary.histogram('dense_output_insulin', dense_output_insulin)
		tf.summary.histogram('labels_insulin', labels_insulin)

	if outprint:
		print()
		print('dense_output_insulin')
		print(dense_output_insulin)
		print()
	
	return loss_insulin, dense_output_insulin

def get_loss_DI(dense_output_DI, labels):

	with tf.name_scope('DI_rmse_loss'):
		dense_output_DI = tf.reshape(dense_output_DI, [batch_size,1])

		labels_DI = tf.reshape(labels[:,13], [batch_size, 1])
		loss_DI = tf.losses.mean_squared_error(labels_DI, dense_output_DI, scope=None, loss_collection=tf.GraphKeys.LOSSES)#, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS)
		loss_DI = tf.squeeze(tf.reduce_mean(tf.sqrt(loss_DI)))

		tf.summary.scalar('loss_DI', loss_DI)
		tf.summary.histogram('dense_output_DI', dense_output_DI)
		tf.summary.histogram('labels_DI', labels_DI)

	if outprint:
		print()
		print('dense_output_DI')
		print(dense_output_DI)
		print()
	
	return loss_DI, dense_output_DI

def get_loss_diab_cont(dense_output_diab_cont, labels):

	with tf.name_scope('diab_cont_rmse_loss'):
		dense_output_diab_cont = tf.reshape(dense_output_diab_cont, [batch_size,1])

		labels_diab_cont = tf.reshape(labels[:,10], [batch_size, 1])
		loss_diab_cont = tf.losses.mean_squared_error(labels_diab_cont, dense_output_diab_cont, scope=None, loss_collection=tf.GraphKeys.LOSSES)#, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS)
		loss_diab_cont = tf.squeeze(tf.reduce_mean(tf.sqrt(loss_diab_cont)))

		tf.summary.scalar('loss_diab_cont', loss_diab_cont)
		tf.summary.histogram('dense_output_diab_cont', dense_output_diab_cont)
		tf.summary.histogram('labels_diab_cont', labels_diab_cont)

	if outprint:
		print()
		print('dense_output_diab_cont')
		print(dense_output_diab_cont)
		print()
	
	return loss_diab_cont, dense_output_diab_cont

def get_loss_diab_all(dense_output_diab_all, labels):

	with tf.name_scope('diab_cross_entropy_loss'):

		dense_output_diab_all = tf.reshape(dense_output_diab_all, [batch_size,2])

		labels_diab_all = tf.cast(tf.reshape(labels[:,12], [batch_size, 1]), tf.int32)
		negative_diab_all = tf.cast((1 - labels_diab_all), tf.int32)

		labels_diab_all2D = tf.concat([labels_diab_all, negative_diab_all], 1)

		# loss_diab_all = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_diab_all2D, logits=dense_output_diab_all))
		loss_diab_all = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels_diab_all2D, logits=dense_output_diab_all))

		preds_diab_all = tf.nn.softmax(dense_output_diab_all)

		loss_true_positives = tf.reduce_mean(tf.cast(labels_diab_all, tf.float32) * preds_diab_all[:,0])
		loss_true_negatives = tf.reduce_mean(tf.cast(negative_diab_all, tf.float32) * preds_diab_all[:,1])


		# f1_tp = lambda: 0.
		# f2_tp = lambda: 15. * (1. - loss_true_positives)
		# loss_tp = tf.case([(tf.greater(loss_true_positives, 0.), f2_tp)], default=f1_tp)

		# f1_tn = lambda: 0.
		# f2_tn = lambda: 1. - loss_true_negatives
		# loss_tn = tf.case([(tf.greater(loss_true_negatives, 0.), f2_tn)], default=f1_tn)


		# full_loss_diab_all = loss_diab_all + loss_tp + loss_tn

		full_loss_diab_all = loss_diab_all

		tf.summary.scalar('loss_diab_all', loss_diab_all)
		# tf.summary.scalar('loss_tp', loss_tp)
		# tf.summary.scalar('loss_tn', loss_tn)
		tf.summary.scalar('full_loss_diab_all', full_loss_diab_all)
		tf.summary.histogram('dense_output_diab_all', dense_output_diab_all)
		tf.summary.histogram('preds_diab_all', preds_diab_all)
		tf.summary.histogram('preds_diab_all_vec', preds_diab_all[:,0])
		tf.summary.histogram('labels_diab_all', labels_diab_all2D)
		tf.summary.histogram('labels_diab_all_vec', labels_diab_all2D[:,0])


	if outprint:
		printer(dense_output_diab_all, 'dense_output_diab_all')
		printer(preds_diab_all, 'preds_diab_all')
		

	return full_loss_diab_all, tf.reshape(preds_diab_all[:,0], [batch_size,1])

def get_loss_all(dense_output_all, labels):

	with tf.name_scope('regression_losses'):

		dense_output_all = tf.reshape(dense_output_all, [batch_size,10])
		tf.summary.histogram('dense_output_all', dense_output_all)

		# cont labels
		age_label = tf.reshape(labels[:,1], [-1])
		age_loss = tf.losses.mean_squared_error(age_label, dense_output_all[:,0], scope=None, loss_collection=tf.GraphKeys.LOSSES)#, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS)
		age_loss = tf.squeeze(tf.reduce_mean(tf.sqrt(age_loss)))
		age_out = dense_output_all[:,0]
		tf.summary.scalar('age_loss', age_loss)
		tf.summary.histogram('age_out', age_out)

		bmi_label = tf.reshape(labels[:,4], [-1])
		bmi_loss = tf.losses.mean_squared_error(bmi_label, dense_output_all[:,1], scope=None, loss_collection=tf.GraphKeys.LOSSES)#, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS)
		bmi_loss = tf.squeeze(tf.reduce_mean(tf.sqrt(bmi_loss)))
		bmi_out = dense_output_all[:,3]
		tf.summary.scalar('bmi_loss', bmi_loss)
		tf.summary.histogram('bmi_out', bmi_out)

		fatimp_label = tf.reshape(labels[:,1], [-1])
		fatimp_loss = tf.losses.mean_squared_error(fatimp_label, dense_output_all[:,2], scope=None, loss_collection=tf.GraphKeys.LOSSES)#, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS)
		fatimp_loss = tf.squeeze(tf.reduce_mean(tf.sqrt(fatimp_loss)))
		fatimp_out = dense_output_all[:,4]
		tf.summary.scalar('fatimp_loss', fatimp_loss)
		tf.summary.histogram('fatimp_out', fatimp_out)

		insulin_label = tf.reshape(labels[:,6], [-1])
		insulin_loss = tf.losses.mean_squared_error(insulin_label, dense_output_all[:,3], scope=None, loss_collection=tf.GraphKeys.LOSSES)#, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS)
		insulin_loss = tf.squeeze(tf.reduce_mean(tf.sqrt(insulin_loss)))
		insulin_out = dense_output_all[:,5]
		tf.summary.scalar('insulin_loss', insulin_loss)
		tf.summary.histogram('insulin_out', insulin_out)

		di_label = tf.reshape(labels[:,10], [-1])
		di_loss = tf.losses.mean_squared_error(di_label, dense_output_all[:,4], scope=None, loss_collection=tf.GraphKeys.LOSSES)#, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS)
		di_loss = tf.squeeze(tf.reduce_mean(tf.sqrt(di_loss)))
		di_out = dense_output_all[:,6]
		tf.summary.scalar('di_loss', di_loss)
		tf.summary.histogram('di_out', di_out)

		hba1c_label = tf.reshape(labels[:,13], [-1])
		hba1c_loss = tf.losses.mean_squared_error(hba1c_label, dense_output_all[:,5], scope=None, loss_collection=tf.GraphKeys.LOSSES)#, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS)
		hba1c_loss = tf.squeeze(tf.reduce_mean(tf.sqrt(hba1c_loss)))
		hba1c_out = dense_output_all[:,7]
		tf.summary.scalar('hba1c_loss', hba1c_loss)
		tf.summary.histogram('age_ohba1c_outut', hba1c_out)


	with tf.name_scope('classification_losses'):

		# diabetes 2D labels
		diab_2d_label = tf.cast(tf.reshape(labels[:,12], [batch_size, 1]), tf.int32)
		negative_diab_all = tf.cast((1 - diab_2d_label), tf.int32)
		labels_diab_all2D = tf.concat([diab_2d_label, negative_diab_all], 1)
		diab_loss = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits(
				labels=labels_diab_all2D, 
				logits=tf.concat(
					[
					tf.reshape(dense_output_all[:,6], [-1,1]), 
					tf.reshape(dense_output_all[:,7], [-1,1])
					], 1)
				)
			)
		diab_out2d = tf.nn.softmax(tf.concat(
					[
					tf.reshape(dense_output_all[:,6], [-1,1]), 
					tf.reshape(dense_output_all[:,7], [-1,1])
					], 1))

		diab_out = diab_out2d[:,0]

		tf.summary.histogram('diab_out', diab_out)
		tf.summary.histogram('diab_out2d', diab_out2d)
		tf.summary.scalar('diab_loss', diab_loss)


		
		# sex 2D labels
		sex_2d_label = tf.cast(tf.reshape(labels[:,0], [batch_size, 1]), tf.int32)
		negative_sex_all = tf.cast((1 - sex_2d_label), tf.int32)
		labels_sex_all2D = tf.concat([sex_2d_label, negative_sex_all], 1)
		sex_loss = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits(
				labels=labels_sex_all2D, 
				logits=tf.concat(
					[
					tf.reshape(dense_output_all[:,8], [-1,1]), 
					tf.reshape(dense_output_all[:,9], [-1,1])
					], 1)
				)
			)		
		sex_out2d = tf.nn.softmax(tf.concat(
					[
					tf.reshape(dense_output_all[:,8], [-1,1]), 
					tf.reshape(dense_output_all[:,9], [-1,1])
					], 1))
		sex_out = sex_out2d[:,0]


		tf.summary.histogram('sex_out', sex_out)
		tf.summary.histogram('sex_out2d', sex_out2d)
		tf.summary.scalar('sex_loss', sex_loss)



	with tf.name_scope('final_loss'):

		full_loss = [age_loss, bmi_loss, fatimp_loss, insulin_loss, di_loss, hba1c_loss, diab_loss, sex_loss]

		preds_all = tf.concat([
			tf.reshape(age_out, [-1,1]), 
			tf.reshape(bmi_out, [-1,1]),
			tf.reshape(fatimp_out, [-1,1]),
			tf.reshape(insulin_out, [-1,1]),
			tf.reshape(di_out, [-1,1]),
			tf.reshape(hba1c_out, [-1,1]),
			tf.reshape(diab_out, [-1,1]),
			tf.reshape(sex_out, [-1,1])
			], 1)

		tf.summary.histogram('full_loss_vec', full_loss)
		tf.summary.histogram('full_loss_sum', tf.reduce_sum(full_loss))

		# preds_all = tf.concat([age_out, height_out, weight_out, bmi_out, fatimp_out, insulin_out, di_out, hba1c_out, diab_out, sex_out], axis=1)

	return full_loss, tf.reshape(preds_all, [-1,8])







#################################################################################################
#################################################################################################
#################################################################################################
####################################### data preprocessing ######################################
#################################################################################################
#################################################################################################
#################################################################################################







training_batch_labels, body_array_shape, num_val_samples = Preprocessing()


num_loaded_batches = int((training_batch_labels.shape[0]/(batch_size*train_data_split)))
num_val_batches = int((num_val_samples/batch_size)-1)

if outprint:
	print()
	print('num_loaded_batches.........................', num_loaded_batches)






#################################################################################################
#################################################################################################
#################################################################################################
################################ data iterator and training flag ################################
#################################################################################################
#################################################################################################
#################################################################################################






with tf.name_scope('parameter'):

	training_flag = tf.placeholder(tf.bool, name='training_flag')
	learning_rate = tf.placeholder(tf.float32, name='learning_rate')

	tf.summary.scalar("learning_rate", learning_rate)

with tf.name_scope('data_iterator'):

	scans_placeholder = tf.placeholder(tf.float32, 
									   shape=(None, 
										      body_array_shape[0], 
										      body_array_shape[1], 
										      body_array_shape[2],
										   	  1), 
									   name='train_scans')

	labels_placeholder = tf.placeholder(tf.float32, 
									    shape=(None, len(use_cols)), 
									    name='train_labels')


	dataset = tf.data.Dataset.from_tensor_slices((scans_placeholder, labels_placeholder)
												).shuffle(buffer_size=100
												).batch(batch_size)

	iterator =  dataset.make_initializable_iterator()

	bodies, labels = iterator.get_next()

	print()
	printer(bodies, 'input bodies')
	printer(labels, 'input labels')
	print()






#################################################################################################
#################################################################################################
#################################################################################################
################################# inference and loss computation ################################
#################################################################################################
#################################################################################################
#################################################################################################






with tf.device('/device:CPU:0'):

	with tf.name_scope('get_add_feats'):

		add_input = tf.concat(
			[
			tf.reshape(labels[:,0], [-1,1]), 
			tf.reshape(labels[:,1], [-1,1]), 
			tf.reshape(labels[:,2], [-1,1]), 
			tf.reshape(labels[:,3], [-1,1]), 
			tf.reshape(labels[:,4], [-1,1]), 
			tf.reshape(labels[:,5], [-1,1]), 
			tf.reshape(labels[:,7], [-1,1]), 
			tf.reshape(labels[:,8], [-1,1]), 
			tf.reshape(labels[:,9], [-1,1])
			], axis=1)


with tf.device('/device:GPU:0'):

	if nn_type == 'dense_net':
		dense_output0, grad_atts0, x_embedding = DenseNet(x=bodies, add_feats=add_input, filters=growth_k, training=training_flag).model
	if nn_type == 'conv_net':
		dense_output0, grad_atts0 = ConvNet(x=bodies, add_feats=add_input, training=training_flag).model


with tf.device('/device:CPU:0'):

	loss_vec, dense_output = get_loss_all(dense_output0, labels)
	loss = tf.reduce_sum(loss_vec)
	tf.summary.scalar('summed_loss', loss)






#################################################################################################
#################################################################################################
#################################################################################################
################################### visualisation placeholders ##################################
#################################################################################################
#################################################################################################
#################################################################################################





with tf.name_scope('validation_scalars'):


	validation_loss_placeholder_total = tf.placeholder_with_default(0.0, shape=[])
	validation_loss_summary_total = tf.summary.scalar("validation_loss_total", validation_loss_placeholder_total) 

	validation_loss_placeholder_age = tf.placeholder_with_default(0.0, shape=[])
	validation_loss_summary_age = tf.summary.scalar("validation_loss_age", validation_loss_placeholder_age) 

	validation_loss_placeholder_bmi = tf.placeholder_with_default(0.0, shape=[])
	validation_loss_summary_bmi = tf.summary.scalar("validation_loss_bmi", validation_loss_placeholder_bmi) 

	validation_loss_placeholder_fatimp = tf.placeholder_with_default(0.0, shape=[])
	validation_loss_summary_fatimp = tf.summary.scalar("validation_loss_fatimp", validation_loss_placeholder_fatimp) 

	validation_loss_placeholder_insulin = tf.placeholder_with_default(0.0, shape=[])
	validation_loss_summary_insulin = tf.summary.scalar("validation_loss_insulin", validation_loss_placeholder_insulin) 

	validation_loss_placeholder_di = tf.placeholder_with_default(0.0, shape=[])
	validation_loss_summary_di = tf.summary.scalar("validation_loss_di", validation_loss_placeholder_di) 

	validation_loss_placeholder_hba1c = tf.placeholder_with_default(0.0, shape=[])
	validation_loss_summary_hba1c = tf.summary.scalar("validation_loss_hba1c", validation_loss_placeholder_hba1c) 


	validation_loss_placeholder_diab = tf.placeholder_with_default(0.0, shape=[])
	validation_loss_summary_diab = tf.summary.scalar("validation_loss_diab", validation_loss_placeholder_diab) 

	validation_auc_placeholder_diab = tf.placeholder_with_default(0.0, shape=[])
	validation_loss_summary_diab_auc = tf.summary.scalar("validation_auc__diab", validation_auc_placeholder_diab) 

	# validation_precision_placeholder_diab = tf.placeholder_with_default(0.0, shape=[])
	# validation_loss_summary_diab_precision = tf.summary.scalar("validation_precision_diab", validation_precision_placeholder_diab) 


	validation_loss_placeholder_sex = tf.placeholder_with_default(0.0, shape=[])
	validation_loss_summary_sex = tf.summary.scalar("validation_loss_sex", validation_loss_placeholder_sex) 

	validation_auc_placeholder_sex = tf.placeholder_with_default(0.0, shape=[])
	validation_loss_summary_sex_auc = tf.summary.scalar("validation_auc_sex", validation_auc_placeholder_sex) 

	# validation_precision_placeholder_sex = tf.placeholder_with_default(0.0, shape=[])
	# validation_loss_summary_sex_precision = tf.summary.scalar("validation_precision_sex", validation_precision_placeholder_sex) 






#################################################################################################
#################################################################################################
#################################################################################################
#################################### build graph and session ####################################
#################################################################################################
#################################################################################################
#################################################################################################







sys.stdout.flush()

saver = tf.train.Saver(tf.global_variables())


config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True


with tf.Session(config=config) as sess:

	# sess = tf_debug.LocalCLIDebugWrapperSession(sess)

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

	with tf.name_scope('Optimizer'):

		with tf.control_dependencies(update_ops):

			global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

			# optimizer = tf.train.AdamOptimizer(
			# 	learning_rate=learning_rate,
			# 	beta1=0.9,
			# 	beta2=0.999,
			# 	epsilon=1e-08,
			# 	use_locking=False,
			# 	name='Adam')

			optimizer = tf.train.RMSPropOptimizer(
				learning_rate,
				decay=0.99,
				momentum=0.0,
				epsilon=1e-10,
				use_locking=False,
				centered=True,
				name='RMSProp')

			grads_and_vars = optimizer.compute_gradients(loss, grad_loss=None)
			train_step = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(Target_dir + 'train/', sess.graph)
	val_writer = tf.summary.FileWriter(Target_dir + 'val/')

	sess.run(tf.global_variables_initializer())







	#################################################################################################
	#################################################################################################
	#################################################################################################
	######################################## run training set #######################################
	#################################################################################################
	#################################################################################################
	#################################################################################################







	training_cache, validation_cache = [], []
	counter = 0
	lrdummy = init_learning_rate
	epoch_learning_rate = init_learning_rate
	checkpoint_dummy1, checkpoint_dummy2 = 1e+6, 1e-6


	for epoch in range(total_epochs):

		epoch_index = np.random.choice(training_batch_labels.shape[0], size=(num_loaded_batches*batch_size, train_data_split))


		#################################################################################################
		###################################### learning rate decay ######################################
		#################################################################################################


		## learning rate adoption
		if use_cycles == False:
			# if epoch == 5:
			# 	epoch_learning_rate /= 2.
			if epoch % adapt_lr_freq == 0 and epoch != 0:
				epoch_learning_rate *= adapt_lr_ratio
		if use_cycles == True:
			lrdummy *= lr_decay
			epoch_learning_rate = lrdummy + lr_amp * lrdummy * np.cos(.25*epoch*np.pi)



		#################################################################################################
		########################################### run epoch ###########################################
		#################################################################################################


		for partition in range(train_data_split):

			train_log = []


			#################################################################################################
			####################################### load training set #######################################
			#################################################################################################


			with tf.device('/device:CPU:0'):

				input_data, input_labels = load_training_data(epoch_index[:,partition], training_batch_labels)

			
				sess.run(iterator.initializer, 
						 feed_dict = {scans_placeholder: input_data, 
						 			  labels_placeholder: input_labels})


			#################################################################################################
			######################################### run partition #########################################
			#################################################################################################


			for step in range(num_loaded_batches):

				with tf.device('/device:GPU:0'):

					summary, _, run_loss = sess.run([merged, train_step, loss],
												feed_dict = {
													training_flag: True,
													learning_rate: epoch_learning_rate
													})


			
				counter += 1
				train_log.append(run_loss)

				if step % int(np.round(num_loaded_batches/4)) == 0:
					train_writer.add_summary(summary, counter)





			#################################################################################################
			#################################################################################################
			#################################################################################################
			####################################### run validation set ######################################
			#################################################################################################
			#################################################################################################
			#################################################################################################


			embeddings_validation, labels_validation = [], []


			#################################################################################################
			###################################### load validation set ######################################
			#################################################################################################


			with tf.device('/device:CPU:0'):

				input_data, input_labels = load_validation_data()

				sess.run(iterator.initializer, 
						 feed_dict = {scans_placeholder: input_data, 
						 			  labels_placeholder: input_labels})



			for step in range(int((input_labels.shape[0]/batch_size))):


				#################################################################################################
				####################################### run validation set ######################################
				#################################################################################################


				if step == 0:

					val_log_loss = []
					val_log_labels_sex = []
					val_log_labels_age = []
					val_log_labels_bmi = []
					val_log_labels_fatimp = []
					val_log_labels_insulin = []
					val_log_labels_di = []
					val_log_labels_hba1c = []
					val_log_labels_diab = []
					val_log_preds = []
					val_log_bodies = []
					val_log_atts0 = []


				with tf.device('/device:GPU:0'):

					val_loss, val_preds, val_labels, val_bodies, val_atts0, embedding_input = sess.run([
						loss_vec, 
						dense_output, 
						labels,
						bodies,
						grad_atts0,
						x_embedding],
						feed_dict = {training_flag: False})



				embeddings_validation.append(np.asarray(embedding_input))
				labels_validation.append(val_labels)

				val_log_loss.append(val_loss)
				val_log_labels_diab.append(np.reshape(np.asarray(val_labels)[:,12], [-1]))
				val_log_labels_sex.append(np.reshape(np.asarray(val_labels)[:,0], [-1]))
				val_log_labels_age.append(np.reshape(np.asarray(val_labels)[:,1], [-1]))
				val_log_labels_bmi.append(np.reshape(np.asarray(val_labels)[:,4], [-1]))
				val_log_labels_fatimp.append(np.reshape(np.asarray(val_labels)[:,5], [-1]))
				val_log_labels_insulin.append(np.reshape(np.asarray(val_labels)[:,6], [-1]))
				val_log_labels_hba1c.append(np.reshape(np.asarray(val_labels)[:,10], [-1]))
				val_log_labels_di.append(np.reshape(np.asarray(val_labels)[:,13], [-1]))

				val_log_preds.append(np.asarray(val_preds))
				val_log_bodies.append(np.reshape(np.asarray(val_bodies), [-1]))
				val_log_atts0.append(np.reshape(np.asarray(val_atts0), [-1]))


			#################################################################################################
			################################### process validation output ###################################
			#################################################################################################


			if epoch == 0 and partition == 0:
				np.save(Target_dir + 'bodies_shape', val_bodies.shape)
				np.save(Target_dir + 'atts0_shape', val_atts0.shape)


			val_log_loss = np.asarray(val_log_loss)
			val_log_loss = np.reshape(val_log_loss, [-1, val_log_loss.shape[-1]])
			val_log_preds = np.asarray(val_log_preds)
			val_log_preds = np.reshape(val_log_preds, [-1, val_log_preds.shape[-1]])

			val_log_labels_diab = np.reshape(np.asarray(val_log_labels_diab), [-1])
			val_log_labels_sex = np.reshape(np.asarray(val_log_labels_sex), [-1])

			val_log_labels_age = np.reshape(np.asarray(val_log_labels_age), [-1])
			val_log_labels_bmi = np.reshape(np.asarray(val_log_labels_bmi), [-1])
			val_log_labels_fatimp = np.reshape(np.asarray(val_log_labels_fatimp), [-1])
			val_log_labels_insulin = np.reshape(np.asarray(val_log_labels_insulin), [-1])
			val_log_labels_hba1c = np.reshape(np.asarray(val_log_labels_hba1c), [-1])
			val_log_labels_di = np.reshape(np.asarray(val_log_labels_di), [-1])


			embeddings_validation = np.asarray(embeddings_validation)
			embeddings_validation = np.reshape(embeddings_validation, [-1, embeddings_validation.shape[-1]])
			labels_validation = np.asarray(labels_validation)
			labels_validation = np.reshape(labels_validation, [-1, labels_validation.shape[-1]])


			area_roc_hba1c = compute_auc(val_log_preds[:,5], np.reshape(np.asarray(val_log_labels_diab), [-1]))
			area_roc_diab = compute_auc(val_log_preds[:,6], np.reshape(np.asarray(val_log_labels_diab), [-1]))
			area_roc_sex = compute_auc(val_log_preds[:,7], np.reshape(np.asarray(val_log_labels_sex), [-1]))
			precision_hba1c = compute_precision_recall(val_log_preds[:,5], np.reshape(np.asarray(val_log_labels_diab), [-1]))
			precision_diab = compute_precision_recall(val_log_preds[:,6], np.reshape(np.asarray(val_log_labels_diab), [-1]))
			precision_sex = compute_precision_recall(val_log_preds[:,7], np.reshape(np.asarray(val_log_labels_sex), [-1]))

			expvar_age = compute_explained_variance(val_log_preds[:,0], np.reshape(np.asarray(val_log_labels_age), [-1]))
			expvar_bmi = compute_explained_variance(val_log_preds[:,1], np.reshape(np.asarray(val_log_labels_bmi), [-1]))
			expvar_fatimp = compute_explained_variance(val_log_preds[:,2], np.reshape(np.asarray(val_log_labels_fatimp), [-1]))
			expvar_insulin = compute_explained_variance(val_log_preds[:,3], np.reshape(np.asarray(val_log_labels_insulin), [-1]))
			expvar_hba1c = compute_explained_variance(val_log_preds[:,4], np.reshape(np.asarray(val_log_labels_hba1c), [-1]))
			expvar_di = compute_explained_variance(val_log_preds[:,5], np.reshape(np.asarray(val_log_labels_di), [-1]))


			#################################################################################################
			################################ write validation to tensorboard ################################
			#################################################################################################


			validation_summary_total, validation_summary_age, validation_summary_bmi, validation_summary_fatimp, validation_summary_insulin, validation_summary_di, validation_summary_hba1c, validation_summary_diab, validation_summary_sex, validation_summary_diab_auc, validation_summary_sex_auc = sess.run(
				[
				validation_loss_summary_total, 
				validation_loss_summary_age, 
				validation_loss_summary_bmi, 
				validation_loss_summary_fatimp, 
				validation_loss_summary_insulin, 
				validation_loss_summary_di, 
				validation_loss_summary_hba1c, 
				validation_loss_summary_diab, 
				validation_loss_summary_sex, 
				validation_loss_summary_diab_auc, 
				validation_loss_summary_sex_auc,
				# validation_loss_summary_diab_precision, 
				# validation_loss_summary_sex_precision
				], 
				feed_dict = {
					validation_loss_placeholder_total: np.mean(val_log_loss[:,0]) + np.mean(val_log_loss[:,1]) + np.mean(val_log_loss[:,2]) + np.mean(val_log_loss[:,3]) + np.mean(val_log_loss[:,4]) + np.mean(val_log_loss[:,5]) + np.mean(val_log_loss[:,6]) + np.mean(val_log_loss[:,7]),
					validation_loss_placeholder_age: np.mean(val_log_loss[:,0]),
					validation_loss_placeholder_bmi: np.mean(val_log_loss[:,1]),
					validation_loss_placeholder_fatimp: np.mean(val_log_loss[:,2]),
					validation_loss_placeholder_insulin: np.mean(val_log_loss[:,3]),
					validation_loss_placeholder_di: np.mean(val_log_loss[:,4]),
					validation_loss_placeholder_hba1c: np.mean(val_log_loss[:,5]),
					validation_loss_placeholder_diab: np.mean(val_log_loss[:,6]),
					validation_loss_placeholder_sex: np.mean(val_log_loss[:,7]),
					validation_auc_placeholder_diab: area_roc_diab,
					validation_auc_placeholder_sex: area_roc_sex,
					# validation_precision_placeholder_diab: np.mean(precision_diab),
					# validation_precision_placeholder_sex: np.mean(precision_sex)
					})
			val_writer.add_summary(validation_summary_total, counter)
			val_writer.add_summary(validation_summary_age, counter)
			val_writer.add_summary(validation_summary_bmi, counter)
			val_writer.add_summary(validation_summary_fatimp, counter)
			val_writer.add_summary(validation_summary_insulin, counter)
			val_writer.add_summary(validation_summary_di, counter)
			val_writer.add_summary(validation_summary_hba1c, counter)
			val_writer.add_summary(validation_summary_diab, counter)
			val_writer.add_summary(validation_summary_sex, counter)
			val_writer.add_summary(validation_summary_diab_auc, counter)
			val_writer.add_summary(validation_summary_sex_auc, counter)
			# val_writer.add_summary(validation_loss_summary_diab_precision, counter)
			# val_writer.add_summary(validation_loss_summary_sex_precision, counter)





			#################################################################################################
			#################################################################################################
			#################################################################################################
			############################### if new highsore on validation set ###############################
			#################################################################################################
			#################################################################################################
			#################################################################################################




			if np.mean(np.asarray(val_log_loss)) < checkpoint_dummy1 or area_roc_diab > checkpoint_dummy2:




				#################################################################################################
				##################################### save validation logs ######################################
				#################################################################################################



				
				checkpoint_dir = make_folder(epoch, partition)

				np.save(checkpoint_dir + '/predictions_validation', val_log_preds)
				np.save(checkpoint_dir + '/labels', val_log_labels_diab)
				np.save(checkpoint_dir + '/bodies', val_log_bodies)
				np.save(checkpoint_dir + '/atts0', val_log_atts0)
				np.save(checkpoint_dir + '/embeddings_validation', embeddings_validation)
				np.save(checkpoint_dir + '/labels_validation', labels_validation)

				if np.mean(np.asarray(val_log_loss)) < checkpoint_dummy1:
					checkpoint_dummy1 = np.mean(np.asarray(val_log_loss))
				if area_roc_diab > checkpoint_dummy2:
					checkpoint_dummy2 = area_roc_diab

				saver.save(sess=sess, save_path=Target_dir + 'model/dense.ckpt')

				
				if outprint:
					print()
					print('current highscore after epoch', epoch, '/partition', partition)
					print('----------------------------------------------------')
					print('total loss........................', np.round(np.mean(np.asarray(val_log_loss)),2))
					print('age')
					print('.............................RMSE:', np.round(np.mean(val_log_loss[:,0]),2))
					print('....................explained_var:', np.round(100*expvar_age,2), '%')
					print('bmi')
					print('.............................RMSE:', np.round(np.mean(val_log_loss[:,1]),2))
					print('....................explained_var:', np.round(100*expvar_bmi,2), '%')
					print('fatimp')
					print('.............................RMSE:', np.round(np.mean(val_log_loss[:,2]),2))
					print('....................explained_var:', np.round(100*expvar_fatimp,2), '%')
					print('insulin', )
					print('.............................RMSE:', np.round(np.mean(val_log_loss[:,3]),2))
					print('....................explained_var:', np.round(100*expvar_insulin,2), '%')
					print('di')
					print('.............................RMSE:', np.round(np.mean(val_log_loss[:,4]),2))
					print('....................explained_var:', np.round(100*expvar_di,2), '%')
					print('hba1c')
					print('.............................RMSE:', np.round(np.mean(val_log_loss[:,5]),2))
					print('....................explained_var:', np.round(expvar_hba1c,2))
					print('..........................AUC_ROC:', np.round(100*area_roc_hba1c,2), '%')
					print('........................precision:', np.round(100*precision_hba1c,2), '%')
					print('.......................')
					print('diabetes')
					print('.....................CrossEntropy:', np.round(np.mean(val_log_loss[:,6]),2))
					print('..........................AUC_ROC:', np.round(100*area_roc_diab,2), '%')
					print('........................precision:', np.round(100*precision_diab,2), '%')
					print('sex')
					print('.....................CrossEntropy:', np.round(np.mean(val_log_loss[:,7]),2))
					print('..........................AUC_ROC:', np.round(100*area_roc_sex,2), '%')
					print('........................precision:', np.round(100*precision_sex,2), '%')
					for spacer in range(4): print()

					sys.stdout.flush()



				#################################################################################################
				#################################################################################################
				#################################################################################################
				######################################### run test set ##########################################
				#################################################################################################
				#################################################################################################
				#################################################################################################




				embeddings_test, labels_test = [], []


				#################################################################################################
				######################################### load test set #########################################
				#################################################################################################


				with tf.device('/device:CPU:0'):

					input_data, input_labels = load_test_data()

					sess.run(iterator.initializer, 
							 feed_dict = {scans_placeholder: input_data, 
							 			  labels_placeholder: input_labels})



				for step in range(int((input_labels.shape[0]/batch_size))):


					#################################################################################################
					########################################## run test set #########################################
					#################################################################################################


					if step == 0:

						val_log_loss = []
						val_log_labels_sex = []
						val_log_labels_age = []
						val_log_labels_bmi = []
						val_log_labels_fatimp = []
						val_log_labels_insulin = []
						val_log_labels_di = []
						val_log_labels_hba1c = []
						val_log_labels_diab = []
						val_log_preds = []
						val_log_bodies = []
						val_log_atts0 = []



					with tf.device('/device:GPU:0'):

						val_loss, val_loss_vec, val_preds, val_labels, val_bodies, val_atts0, embedding_input = sess.run([
							loss, 
							loss_vec, 
							dense_output, 
							labels,
							bodies,
							grad_atts0,
							x_embedding],
							feed_dict = {training_flag: False})


					embeddings_test.append(np.asarray(embedding_input))
					labels_test.append(val_labels)


					val_log_loss.append(val_loss)
					val_log_labels_diab.append(np.reshape(np.asarray(val_labels)[:,12], [-1]))
					val_log_labels_sex.append(np.reshape(np.asarray(val_labels)[:,0], [-1]))
					val_log_labels_age.append(np.reshape(np.asarray(val_labels)[:,1], [-1]))
					val_log_labels_bmi.append(np.reshape(np.asarray(val_labels)[:,4], [-1]))
					val_log_labels_fatimp.append(np.reshape(np.asarray(val_labels)[:,5], [-1]))
					val_log_labels_insulin.append(np.reshape(np.asarray(val_labels)[:,6], [-1]))
					val_log_labels_hba1c.append(np.reshape(np.asarray(val_labels)[:,10], [-1]))
					val_log_labels_di.append(np.reshape(np.asarray(val_labels)[:,13], [-1]))

					val_log_preds.append(np.asarray(val_preds))
					val_log_bodies.append(np.reshape(np.asarray(val_bodies), [-1]))
					val_log_atts0.append(np.reshape(np.asarray(val_atts0), [-1]))


				#################################################################################################
				###################################### process test output ######################################
				#################################################################################################


				val_log_loss = np.asarray(val_log_loss)
				val_log_loss = np.reshape(val_log_loss, [-1, val_log_loss.shape[-1]])
				val_log_preds = np.asarray(val_log_preds)
				val_log_preds = np.reshape(val_log_preds, [-1, val_log_preds.shape[-1]])

				val_log_labels_diab = np.reshape(np.asarray(val_log_labels_diab), [-1])
				val_log_labels_sex = np.reshape(np.asarray(val_log_labels_sex), [-1])

				val_log_labels_age = np.reshape(np.asarray(val_log_labels_age), [-1])
				val_log_labels_bmi = np.reshape(np.asarray(val_log_labels_bmi), [-1])
				val_log_labels_fatimp = np.reshape(np.asarray(val_log_labels_fatimp), [-1])
				val_log_labels_insulin = np.reshape(np.asarray(val_log_labels_insulin), [-1])
				val_log_labels_hba1c = np.reshape(np.asarray(val_log_labels_hba1c), [-1])
				val_log_labels_di = np.reshape(np.asarray(val_log_labels_di), [-1])


				embeddings_test = np.asarray(embeddings_test)
				embeddings_test = np.reshape(embeddings_test, [-1, embeddings_test.shape[-1]])
				labels_test = np.asarray(labels_test)
				labels_test = np.reshape(labels_test, [-1, labels_test.shape[-1]])

				np.save(checkpoint_dir + '/embeddings_test', embeddings_test)
				np.save(checkpoint_dir + '/labels_test', labels_test)
				np.save(checkpoint_dir + '/predictions_test', val_log_preds)


				area_roc_diab = compute_auc(val_log_preds[:,6], np.reshape(np.asarray(val_log_labels_diab), [-1]))
				area_roc_sex = compute_auc(val_log_preds[:,7], np.reshape(np.asarray(val_log_labels_sex), [-1]))
				precision_diab = compute_precision_recall(val_log_preds[:,6], np.reshape(np.asarray(val_log_labels_sex), [-1]))
				precision_sex = compute_precision_recall(val_log_preds[:,7], np.reshape(np.asarray(val_log_labels_sex), [-1]))

				expvar_age = compute_explained_variance(val_log_preds[:,0], np.reshape(np.asarray(val_log_labels_age), [-1]))
				expvar_bmi = compute_explained_variance(val_log_preds[:,1], np.reshape(np.asarray(val_log_labels_bmi), [-1]))
				expvar_fatimp = compute_explained_variance(val_log_preds[:,2], np.reshape(np.asarray(val_log_labels_fatimp), [-1]))
				expvar_insulin = compute_explained_variance(val_log_preds[:,3], np.reshape(np.asarray(val_log_labels_insulin), [-1]))
				expvar_hba1c = compute_explained_variance(val_log_preds[:,4], np.reshape(np.asarray(val_log_labels_hba1c), [-1]))
				expvar_di = compute_explained_variance(val_log_preds[:,5], np.reshape(np.asarray(val_log_labels_di), [-1]))



				area_roc_hba1c = compute_auc(val_log_preds[:,5], np.reshape(np.asarray(val_log_labels_diab), [-1]))
				area_roc_diab = compute_auc(val_log_preds[:,6], np.reshape(np.asarray(val_log_labels_diab), [-1]))
				area_roc_sex = compute_auc(val_log_preds[:,7], np.reshape(np.asarray(val_log_labels_sex), [-1]))
				precision_hba1c = compute_precision_recall(val_log_preds[:,5], np.reshape(np.asarray(val_log_labels_diab), [-1]))
				precision_diab = compute_precision_recall(val_log_preds[:,6], np.reshape(np.asarray(val_log_labels_diab), [-1]))
				precision_sex = compute_precision_recall(val_log_preds[:,7], np.reshape(np.asarray(val_log_labels_sex), [-1]))

				expvar_age = compute_explained_variance(val_log_preds[:,0], np.reshape(np.asarray(val_log_labels_age), [-1]))
				expvar_bmi = compute_explained_variance(val_log_preds[:,1], np.reshape(np.asarray(val_log_labels_bmi), [-1]))
				expvar_fatimp = compute_explained_variance(val_log_preds[:,2], np.reshape(np.asarray(val_log_labels_fatimp), [-1]))
				expvar_insulin = compute_explained_variance(val_log_preds[:,3], np.reshape(np.asarray(val_log_labels_insulin), [-1]))
				expvar_hba1c = compute_explained_variance(val_log_preds[:,4], np.reshape(np.asarray(val_log_labels_hba1c), [-1]))
				expvar_di = compute_explained_variance(val_log_preds[:,5], np.reshape(np.asarray(val_log_labels_di), [-1]))


				#################################################################################################
				####################################### print test results ######################################
				#################################################################################################

		
				if outprint:
					print()
					print('performance on test set')
					print('----------------------------------------------------')
					print('total loss........................', np.round(np.mean(np.asarray(val_log_loss)),2))
					print('age')
					print('.............................RMSE:', np.round(np.mean(val_log_loss[:,0]),2))
					print('....................explained_var:', np.round(100*expvar_age,2), '%')
					print('bmi')
					print('.............................RMSE:', np.round(np.mean(val_log_loss[:,1]),2))
					print('....................explained_var:', np.round(100*expvar_bmi,2), '%')
					print('fatimp')
					print('.............................RMSE:', np.round(np.mean(val_log_loss[:,2]),2))
					print('....................explained_var:', np.round(100*expvar_fatimp,2), '%')
					print('insulin', )
					print('.............................RMSE:', np.round(np.mean(val_log_loss[:,3]),2))
					print('....................explained_var:', np.round(100*expvar_insulin,2), '%')
					print('di')
					print('.............................RMSE:', np.round(np.mean(val_log_loss[:,4]),2))
					print('....................explained_var:', np.round(100*expvar_di,2), '%')
					print('hba1c')
					print('.............................RMSE:', np.round(np.mean(val_log_loss[:,5]),2))
					print('....................explained_var:', np.round(expvar_hba1c,2))
					print('..........................AUC_ROC:', np.round(100*area_roc_hba1c,2), '%')
					print('........................precision:', np.round(100*precision_hba1c,2), '%')
					print('.......................')
					print('diabetes')
					print('.....................CrossEntropy:', np.round(np.mean(val_log_loss[:,6]),2))
					print('..........................AUC_ROC:', np.round(100*area_roc_diab,2), '%')
					print('........................precision:', np.round(100*precision_diab,2), '%')
					print('sex')
					print('.....................CrossEntropy:', np.round(np.mean(val_log_loss[:,7]),2))
					print('..........................AUC_ROC:', np.round(100*area_roc_sex,2), '%')
					print('........................precision:', np.round(100*precision_sex,2), '%')
					for spacer in range(4): print()

					sys.stdout.flush()










				#################################################################################################
				#################################################################################################
				#################################################################################################
				################################# run training set embeddings ###################################
				#################################################################################################
				#################################################################################################
				#################################################################################################






				embeddings_training, labels_training, val_log_preds = [], [], []


				epoch_index = np.random.choice(training_batch_labels.shape[0], size=(num_loaded_batches*batch_size, train_data_split))


				#################################################################################################
				########################################### run epoch ###########################################
				#################################################################################################


				for partition in range(train_data_split):


					#################################################################################################
					####################################### load training set #######################################
					#################################################################################################


					with tf.device('/device:CPU:0'):

						input_data, input_labels = load_training_data(epoch_index[:,partition], training_batch_labels)

					
						sess.run(iterator.initializer, 
								 feed_dict = {scans_placeholder: input_data, 
								 			  labels_placeholder: input_labels})


					#################################################################################################
					######################################### run partition #########################################
					#################################################################################################


					for step in range(num_loaded_batches):

						with tf.device('/device:GPU:0'):

							val_preds, val_labels, embedding_input = sess.run([
								dense_output, 
								labels,
								x_embedding],
								feed_dict = {training_flag: False})


						embeddings_training.append(np.asarray(embedding_input))
						labels_training.append(val_labels)

						val_log_preds.append(np.asarray(val_preds))


				val_log_preds = np.asarray(val_log_preds)

				val_log_preds = np.reshape(val_log_preds, [-1, val_log_preds.shape[-1]])


				embeddings_training = np.asarray(embeddings_training)
				embeddings_training = np.reshape(embeddings_training, [-1, embeddings_training.shape[-1]])
				labels_training = np.asarray(labels_training)
				labels_training = np.reshape(labels_training, [-1, labels_training.shape[-1]])

				np.save(checkpoint_dir + '/embeddings_training', embeddings_training)
				np.save(checkpoint_dir + '/labels_training', labels_training)
				np.save(checkpoint_dir + '/predictions_training', val_log_preds)














				#################################################################################################
				#################################################################################################
				#################################################################################################
				################################### save arrays for plotting ####################################
				#################################################################################################
				#################################################################################################
				#################################################################################################




				validation_cache.append(np.mean(np.asarray(val_log_loss)))
				training_cache.append(np.mean(np.asarray(train_log)))
				np.save(Target_dir + 'training', np.asarray(training_cache))
				np.save(Target_dir + 'validation', np.asarray(validation_cache))

				





















