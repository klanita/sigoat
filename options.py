import argparse
PATH = '/home/anna.susmelj@ad.biognosys.ch/MIDL/data/'

def parse_args():
    parser = argparse.ArgumentParser(
        description='Adversarial Autoencoder for Domain Adaptation of Optoacoustic signal.')

    parser.add_argument('--test', type=int, default=0)
    
    parser.add_argument('--oldcode', type=int, default=0)

    parser.add_argument('--file_in', type=str,
        default='/home/anna/dlbirhoui_data/parsed_simulated_ellipsesSkinMask_mgt_ms_ring_256_ratio_09_1_20210412.h5')
    parser.add_argument('--geometry', type=str, default="multisegmentCup")

    parser.add_argument('--dataset', type=str, default='Forearm', 
        help='Forearm (default) or Finger.')
    
    parser.add_argument('--target', type=str,
        default='/home/anna/dlbirhoui_data/arm.h5')
    
    # parser.add_argument('--target_modality', type=str, default='ground_truth')
    
    parser.add_argument('--mode', type=str, default="style",
        help="Training mode. Style or sides ( sides | sides_unet | sides_old )")
    parser.add_argument('--normalization', type=str,
        default='instance', help='instance | batch')    
    parser.add_argument('--beta1', type=float, default=0.5,\
        help='Beta1 hyperparam for Adam optimizers.')

    parser.add_argument('--split', type=str, default='left')

    # parameters for training
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--lr', type=float, default=0.001)    
    parser.add_argument('--burnin', type=int, default=100)
    parser.add_argument('--n_steps', type=int, default=5)

    parser.add_argument('--tgt_dir', type=str, 
        default="/home/anna/ResultsSignalDA/",
        help="Target directory to save the model.")
    parser.add_argument('--prefix', type=str, default='',
        help="Prefix for the file to save the model.")

    # parser.add_argument('--nimgs', type=int, default=16)

    parser.add_argument('--pretrained', type=str,
        default='/home/anna/style_results/adv2021-06-21_chunk-1_sides/')
    parser.add_argument('--pretrained_style', type=str, 
        default='/home/anna/style_results/deeplatent2021-06-09_chunk-1/')

    # parameters for the loss
    parser.add_argument('--loss', type=str, default="l1")
    parser.add_argument('--patch', type=bool, default=True,
        help='Set True if you want to use patch loss.')
    parser.add_argument('--n_iters', type=int, default=1,\
        help='Iterations to compute discriminator.')
    parser.add_argument('--num_workers', type=int, default=32,\
        help='Number of workers for data loader.')
        
    parser.add_argument('--n_iters_latent', type=int, default=2,\
        help='Iterations to compute discriminator latent.')
    parser.add_argument('--weight_adv', type=float, default=0.001,\
        help='Weight for adversarial losses')
    parser.add_argument('--weight_adv_latent', type=float, default=0.001,\
        help='Weight for adversarial losses LATENT')
    parser.add_argument('--weight_mmd', type=float, default=0.1,\
        help='Weight for adversarial losses')
    parser.add_argument('--weight_cycle', type=float, default=0.1,\
        help='Weight for cycle consistency losses')
    parser.add_argument('--weight_sides', type=float, default=10.0,\
        help='Weight for sides reconstruction losses in full model')
    parser.add_argument('--weight_grad_adv', type=float, default=0.05,\
        help='Weight for gradient penalty.')
    parser.add_argument('--lr_decay_iters', type=int, default=10,\
        help='Learning rate decay.')
    parser.add_argument('--lr_decay_iters_D', type=int, default=10,\
        help='Learning rate decay.')
    parser.add_argument('--weight_decay', type=float, default=1e-5,\
        help='Weight decay.')
    parser.add_argument('--adv_lr_mult', type=float, default=1e-2,\
        help='Adversarial learning rate multiplier compared to reconstruction.')

    args = parser.parse_args()

    if args.pretrained == 'None':
        args.pretrained = None
    if args.pretrained_style == 'None':
        args.pretrained_style = None

    if args.dataset == 'Finger':
        args.target = '/home/firat/docs/dlbirhoui/parsed_data/parsed_fingerBP_and_signals.h5'
        args.geometry = 'ring'

    args.test = bool(args.test)

    if args.test:
        print('TESTING mode')
        if args.dataset == 'Finger':
            args.target = '/home/anna/OptoAcoustics/data/finger_test.h5'        
            args.file_in = '/home/anna/OptoAcoustics/data/syn_finger_test.h5'            
        else:
            args.file_in = f'{PATH}/test_vn.h5'
            args.target = f'{PATH}/benchmark_invivo.h5'    
            # args.file_in = '/home/anna/OptoAcoustics/data/test_vn.h5'
            # args.target = '/home/anna/OptoAcoustics/data/benchmark_invivo.h5'            

    return args