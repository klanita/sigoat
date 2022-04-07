def save_subset(fname_h5, file, num_images = 50, resolutionXY = 256):
    imsize = [resolutionXY, resolutionXY]
    sigmat_size = [2030, 256]    
    compression_lvl = 9 # pick between 0 and 9
    
    print('creating file: %s' % fname_h5)
    
    modalities_list = ['sigmat_multisegment']

    data = {}
    with h5py.File(fname_h5, 'w', libver='latest') as h5_fh:
        for modality in modalities_list:
            modality_size = list(file[modality][0].shape)
            data[modality] = h5_fh.create_dataset(
                'sigmat_ring', shape=[num_images] + modality_size, 
                dtype=np.float32, chunks=tuple([1] + modality_size),
                compression='gzip', compression_opts=compression_lvl)

    for i_im in range(num_images):
        if i_im % 10 == 0:
            print(i_im)
        with h5py.File(fname_h5, 'a', libver='latest') as h5_fh:
            h5_fh['sigmat_multisegment'][i_im] = file['sigmat_multisegment'][i_im]
