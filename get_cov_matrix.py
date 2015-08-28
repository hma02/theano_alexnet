'''
Load data in parallel with train.py
'''

import time
import math
import yaml
import glob
import numpy as np

import hickle as hkl

def unpack_configs(config, ext_data='.hkl', ext_label='.npy'):
    flag_para_load = config['para_load']

    # Load Training/Validation Filenames and Labels
    train_folder = config['train_folder']
    val_folder = config['val_folder']
    label_folder = config['label_folder']
    train_filenames = sorted(glob.glob(train_folder + '/*' + ext_data))
    val_filenames = sorted(glob.glob(val_folder + '/*' + ext_data))
    train_labels = np.load(label_folder + 'train_labels' + ext_label)
    val_labels = np.load(label_folder + 'val_labels' + ext_label)
    img_mean = np.load(config['mean_file'])
    img_mean = img_mean[:, :, :, np.newaxis].astype('float32')
    return (flag_para_load, 
            train_filenames, val_filenames, train_labels, val_labels, img_mean)


if __name__ == '__main__':

    with open('config.yaml', 'r') as f:
        config = yaml.load(f)
    with open('spec_2gpu.yaml', 'r') as f:
        config = dict(config.items() + yaml.load(f).items())
    

    # UNPACK CONFIGS
    (flag_para_load, train_filenames, val_filenames,
     train_labels, val_labels, img_mean) = unpack_configs(config)

    train_filenames = train_filenames[:]

    batch_size = config['batch_size']
     
    img_size = 256
    n_train_filenames = len(train_filenames)

    print 'batch_size: %d, %d images' % (batch_size,n_train_filenames)

    div_const =  1.0 * img_size * img_size * batch_size * n_train_filenames
    
    
    
    RR = 0.0
    RG = 0.0
    RB = 0.0
    GG = 0.0
    GB = 0.0
    BB = 0.0
    R_mean = 0.0
    G_mean = 0.0
    B_mean = 0.0
    

    for hkl_name in train_filenames:

        # print hkl_name
	      print hkl_name
        data = hkl.load(hkl_name).astype('int64') # c01b (3,256,256,batch_size)
        
        R=data[0,:,:,:].flatten()
        G=data[1,:,:,:].flatten()
        B=data[2,:,:,:].flatten()    
  
  		
  	RR += np.dot(R,R)/div_const
  	RG += np.dot(R,G)/div_const
  	RB += np.dot(R,B)/div_const
  	GG += np.dot(G,G)/div_const
  	GB += np.dot(G,B)/div_const
  	BB += np.dot(B,B)/div_const
  
  	R_mean += np.mean(R)
  	G_mean += np.mean(G)
  	B_mean += np.mean(B)

    

    R_mean /=  n_train_filenames
    G_mean /=  n_train_filenames
    B_mean /=  n_train_filenames

    print RR,RG,RB,GG,GB,BB,R_mean,G_mean,B_mean

    RR = RR - R_mean*R_mean
    RG = RG - R_mean*G_mean
    RB = RB - R_mean*B_mean
    GG = GG - G_mean*G_mean
    GB = GB - G_mean*B_mean
    BB = BB - B_mean*B_mean

# symmetrical, so just calculate 6 elements
#
#		                sum(R*R)/N-rr   sum(R*G)/N-rg	  sum(R*B)/N-rb
#		    
#   RGB_Cov =   		                sum(G*G)/N-gg	  sum(G*B)/N-gb
#				
#						                                        sum(B*B)/N-bb
#
#
    RGB_Cov = np.asarray([[RR,RG,RB],
			  [RG,GG,GB],
			  [RB,GB,BB]])
    print RGB_Cov

    np.save('./RGB_Cov_matrix.npy',RGB_Cov)
    np.save('./RGB_mean.npy', [R_mean,G_mean,B_mean])
