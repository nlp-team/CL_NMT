import argparse
import torch
import os
from torch.serialization import default_restore_location
from projection import setup_othermodels_PCA_directions
from projection import project_othermodels_trajectory
import plot_2D


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot optimization trajectory')
    parser.add_argument('--model_folder', default='fairseq_master/checkpoints/transformer-iwslt14-de-en-sgd/', help='folders for checkpoints to be projected')
    parser.add_argument('--ignore_embedding', action='store_true',default=False, help='ignore the embedding')
    parser.add_argument('--ignore', default='', help='ignore bias and BN paras: biasbn (no bias or bn)')
    parser.add_argument('--start_epoch', default=1, type=int, help='min index of epochs')
    parser.add_argument('--max_epoch', default=50, type=int, help='max number of epochs')
    parser.add_argument('--save_epoch', default=1, type=int, help='save models every few epochs')
    parser.add_argument('--dir_file', default='', help='load the direction file for projection')

    args = parser.parse_args()

    # load the state of the final  checkpoint
    model_file = args.model_folder + 'checkpoint' +str(args.max_epoch)+'.pt'
    state = torch.load(model_file, map_location=lambda s, l: default_restore_location(s, 'cpu'), )


    # get the args of the transformer
    args_transformer = state['args']

    # update the data path
    args_transformer.data = 'fairseq_master/' + args_transformer.data

    # change the tensorboard_logdir to ''
    args_transformer.tensorboard_logdir = ''

    # get the name and parameter of the model
    s = state['model']
    w = []   # weight of the model

    if args.ignore_embedding:
        for key in s:
            if 'version' in key or '_float_tensor' in key or 'embed' in key:
                s[key].fill_(0)
            w.append(s[key])
    else:
        for key in s:
            if 'version' in key or '_float_tensor' in key:
                s[key].fill_(0)
            w.append(s[key])


    # get the all checkpoints path
    model_files = []
    for epoch in range(args.start_epoch, args.max_epoch+args.save_epoch,args.save_epoch):
        temp_model_file = args.model_folder + 'checkpoint' + str(epoch) +'.pt'
        assert os.path.exists(temp_model_file), 'model %s does not exist' % temp_model_file
        model_files.append(temp_model_file)

    # --------------------------------------------------------------------------
    # load or create projection directions
    # --------------------------------------------------------------------------
    if args.dir_file:
        dir_file = args.dir_file
    else:
        args_transformer.ignore = args.ignore
        dir_file = setup_othermodels_PCA_directions(args_transformer, args.ignore_embedding, model_files, w)

    # get the trajectory file
    proj_file = project_othermodels_trajectory(dir_file, w, model_files, args.ignore_embedding)

    # draw the pdf to the proj_file
    plot_2D.plot_trajectory(proj_file, dir_file)


