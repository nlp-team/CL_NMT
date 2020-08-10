import argparse
import torch
from torch.serialization import default_restore_location
import h5_util
import h5py
import os
from fairseq import checkpoint_utils, distributed_utils, options, progress_bar, tasks, utils
from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter
import numpy as np
import projection as proj
import scheduler
import time
import copy
from train import train, validate
import collections
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import cm
import seaborn as sns


def create_random_direction(states,ignore_embedding,ignore):
    """
        Produce a random direction that is a list of random Gaussian tensors
        with the same shape as the network's state_dict(), so one direction entry
        per weight, including BN's running_mean/var.
    """
    direction = [torch.randn(w.size()) for k, w in states.items()]
    for d, (k, w) in zip(direction, states.items()):
        if (d.dim() <= 1 and ignore=='biasbn') or 'version' in k or '_float_tensor' in k or ('embed' in k and ignore_embedding):
            d.fill_(0)
        else:
            # filter norm
            for di, wi in zip(d, w):
                di.mul_(wi.norm()/(di.norm() + 1e-10))
    return direction

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plotting loss surface')
    parser.add_argument('--model_folder', default='fairseq_master/checkpoints/transformer-iwslt14-de-en-sgd/', help='folders for checkpoints to be projected')
    parser.add_argument('--ignore_embedding', action='store_true', default=False, help='ignore the embedding')
    parser.add_argument('--ignore', default='', help='ignore bias and BN paras: biasbn (no bias or bn)')
    parser.add_argument('--start_epoch', default=1, type=int, help='min index of epochs')
    parser.add_argument('--max_epoch', default=50, type=int, help='max number of epochs')
    parser.add_argument('--save_epoch', default=1, type=int, help='save models every few epochs')
    parser.add_argument('--dir_file', default='', help='load the direction file for projection')
    parser.add_argument('--x', default='-1:1:51', help='A string with format xmin:x_max:xnum')
    parser.add_argument('--y', default=None, help='A string with format ymin:ymax:ynum')

    parser.add_argument('--surf_file', default='',
                        help='customize the name of surface file, could be an existing file.')

    # plot parameters
    parser.add_argument('--proj_file', default='', help='the .h5 file contains projected optimization trajectory.')
    parser.add_argument('--loss_max', default=5, type=float, help='Maximum value to show in 1D plot')
    parser.add_argument('--vmax', default=10, type=float, help='Maximum value to map')
    parser.add_argument('--vmin', default=0.1, type=float, help='Miminum value to map')
    parser.add_argument('--vlevel', default=0.5, type=float, help='plot contours every vlevel')
    parser.add_argument('--show', action='store_true', default=False, help='show plotted figures')
    parser.add_argument('--log', action='store_true', default=False, help='use log scale for loss values')
    parser.add_argument('--plot', action='store_true', default=False, help='plot figures after computation')

    args = parser.parse_args()

    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # load the state of the final  checkpoint
    model_file = args.model_folder + 'checkpoint' + str(args.max_epoch) + '.pt'
    state = torch.load(model_file, map_location=lambda s, l: default_restore_location(s, 'cpu'), )

    # get the args of the transformer
    args_transformer = state['args']

    # update the data path
    args_transformer.data = 'fairseq_master/' + args_transformer.data

    # change the tensorboard_logdir to ''
    args_transformer.tensorboard_logdir = ''


    ## build the translation task
    task = tasks.setup_task(args_transformer)

    ## get the model and criterion
    model = task.build_model(args_transformer)
    criterion = task.build_criterion(args_transformer)

    # states = model.state_dict()
    states = state['model']

    model.load_state_dict(states)

    # --------------------------------------------------------------------------
    # Check plotting resolution
    # --------------------------------------------------------------------------
    try:
        args.xmin, args.xmax, args.xnum = [float(a) for a in args.x.split(':')]
        args.ymin, args.ymax, args.ynum = (None, None, None)
        if args.y:
            args.ymin, args.ymax, args.ynum = [float(a) for a in args.y.split(':')]
            assert args.ymin and args.ymax and args.ynum, \
                'You specified some arguments for the y axis, but not all'
    except:
        raise Exception('Improper format for x- or y-coordinates. Try something like -1:1:51')

    if args.dir_file:
        # if exist the PCA directions
        f = h5py.File(args.dir_file, 'r')
        xdirection = h5_util.read_list(f, 'xdirection')
        ydirection = h5_util.read_list(f, 'ydirection')

    else:
        # build the random directions
        folder_name = 'fairseq_master' + '/' + args.save_dir + '/Random_'
        xdirection = create_random_direction(states,args.ignore_embedding,args.ignore)
        ydirection = create_random_direction(states,args.ignore_embedding,args.ignore)

        folder_name += 'lr=' + str(args.lr[0])
        folder_name += '_optimier=' + str(args.optimizer)
        folder_name += '_ignore_embedding=' + str(args.ignore_embedding)
        if args.ignore:
            folder_name += '_ignoreBN'

        os.system('mkdir ' + folder_name)
        dir_file = folder_name + '/directions.h5'

        f = h5py.File(dir_file, 'w')  # create file, fail if exists

        h5_util.write_list(f, 'xdirection', xdirection)
        h5_util.write_list(f, 'ydirection', ydirection)
        f.close()
        print("direction file created: %s" % dir_file)

    # get the trainer
    args_transformer.restore_file  = 'checkpoint'+str(args.max_epoch)+'.pt'
    trainer = Trainer(args_transformer, task, model, criterion)

    args_transformer.save_dir = 'fairseq_master/' + args_transformer.save_dir

    ## get the surf file path
    surf_file = args.dir_file + '_surf'
    surf_file += '_[%s,%s,%d]' % (str(args.xmin), str(args.xmax), int(args.xnum))
    surf_file += 'x[%s,%s,%d]' % (str(args.ymin), str(args.ymax), int(args.ynum))
    surf_file += '.h5'

    f = h5py.File(surf_file, 'a')
    f['dir_file'] = args.dir_file


    # Create the coordinates(resolutions) at which the function is evaluated
    xcoordinates = np.linspace(args.xmin, args.xmax, num=args.xnum)
    f['xcoordinates'] = xcoordinates

    ycoordinates = np.linspace(args.ymin, args.ymax, num=args.ynum)
    f['ycoordinates'] = ycoordinates
    f.close()

    directions = [xdirection, ydirection]

    similarity = proj.cal_angle(proj.nplist_to_tensor(directions[0]), proj.nplist_to_tensor(directions[1]))
    print('cosine similarity between x-axis and y-axis: %f' % similarity)

    f = h5py.File(surf_file, 'r+')
    xcoordinates = f['xcoordinates'][:]
    ycoordinates = f['ycoordinates'][:]

    losses = -np.ones(shape=(len(xcoordinates), len(ycoordinates)))
    accuracies = -np.ones(shape=(len(xcoordinates), len(ycoordinates)))

    f['train_loss'] = losses
    f['train_acc'] = accuracies

    inds, coords, inds_nums = scheduler.get_job_indices(losses, xcoordinates, ycoordinates, None)

    start_time = time.time()
    total_sync = 0.0

    xtra_state = trainer.load_checkpoint(
        os.path.join(args_transformer.save_dir, args_transformer.restore_file),
        args_transformer.reset_optimizer,
        args_transformer.reset_lr_scheduler,
        eval(args_transformer.optimizer_overrides),
        reset_meters=args_transformer.reset_meters,
    )

    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args_transformer, trainer)

    update_freq = args_transformer.update_freq[epoch_itr.epoch - 1] if epoch_itr.epoch <= len(args_transformer.update_freq) else \
    args_transformer.update_freq[-1]

    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args_transformer.fix_batches_to_gpus,
        shuffle=(epoch_itr.epoch >= args_transformer.curriculum),
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.build_progress_bar(
        args_transformer, itr, epoch_itr.epoch, no_progress_bar='simple',
    )

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    for count, ind in enumerate(inds):
        # Get the coordinates of the loss value being calculated
        coord = coords[count]
        dx = directions[0]
        dy = directions[1]
        changes = [d0 * coord[0] + d1 * coord[1] for (d0, d1) in zip(dx, dy)]
        new_states = copy.deepcopy(states)
        assert (len(new_states) == len(changes))
        for (k, v), d in zip(new_states.items(), changes):
            d = torch.tensor(d)
            v.add_(d.type(v.type()))

        ## upload the weight
        model.load_state_dict(new_states)

        trainer._model = model

        valid_subsets = ['train']

        loss_start = time.time()

        valid_losses = validate(args_transformer, trainer, task, epoch_itr, valid_subsets)

        loss_compute_time = time.time() - loss_start

        print(str(count + 1) + '  ' + str(coord) + '   train_loss:' + str(valid_losses[0]) + '   time: ' + str(loss_compute_time))
        losses.ravel()[ind] = valid_losses[0]

    f['train_loss'][:] = losses
    f.flush()
    f.close()



    f = h5py.File(surf_file, 'r')
    x = np.array(f['xcoordinates'][:])
    y = np.array(f['ycoordinates'][:])
    X, Y = np.meshgrid(x, y)

    surf_name = 'train_loss'

    if surf_name in f.keys():
        Z = np.array(f[surf_name][:])
    elif surf_name == 'train_err' or surf_name == 'test_err':
        Z = 100 - np.array(f[surf_name][:])
    else:
        print('%s is not found in %s' % (surf_name, surf_file))

    print('------------------------------------------------------------------')
    print('plot_2d_contour')
    print('------------------------------------------------------------------')
    print("loading surface file: " + surf_file)
    print('len(xcoordinates): %d   len(ycoordinates): %d' % (len(x), len(y)))
    print('max(%s) = %f \t min(%s) = %f' % (surf_name, np.max(Z), surf_name, np.min(Z)))
    print(Z)

    if (len(x) <= 1 or len(y) <= 1):
        print('The length of coordinates is not enough for plotting contours')
        exit(0)

    # --------------------------------------------------------------------
    # Plot 2D contours
    # --------------------------------------------------------------------
    fig = plt.figure()
    CS = plt.contour(X, Y, Z, cmap='summer', levels=np.arange(args.vmin, args.vmax, args.vlevel))
    plt.clabel(CS, inline=1, fontsize=8)
    fig.savefig(surf_file + '_' + surf_name + '_2dcontour' + '.pdf', dpi=300,
                bbox_inches='tight', format='pdf')

    fig = plt.figure()
    print(surf_file + '_' + surf_name + '_2dcontourf' + '.pdf')
    CS = plt.contourf(X, Y, Z, cmap='summer', levels=np.arange(args.vmin, args.vmax, args.vlevel))
    fig.savefig(surf_file + '_' + surf_name + '_2dcontourf' + '.pdf', dpi=300,
                bbox_inches='tight', format='pdf')

    # --------------------------------------------------------------------
    # Plot 2D heatmaps
    # --------------------------------------------------------------------
    fig = plt.figure()
    sns_plot = sns.heatmap(Z, cmap='viridis', cbar=True, vmin=args.vmin, vmax=args.vmax,
                           xticklabels=False, yticklabels=False)
    sns_plot.invert_yaxis()
    sns_plot.get_figure().savefig(surf_file + '_' + surf_name + '_2dheat.pdf',
                                  dpi=300, bbox_inches='tight', format='pdf')

    # --------------------------------------------------------------------
    # Plot 3D surface
    # --------------------------------------------------------------------
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.savefig(surf_file + '_' + surf_name + '_3dsurface.pdf', dpi=300,
                bbox_inches='tight', format='pdf')

    f.close()

    # --------------------------------------------------------------------
    # Plot contours
    # --------------------------------------------------------------------
    if args.proj_file:
        surf_name = 'train_loss'

        f = h5py.File(surf_file, 'r')
        x = np.array(f['xcoordinates'][:])
        y = np.array(f['ycoordinates'][:])
        X, Y = np.meshgrid(x, y)
        if surf_name in f.keys():
            Z = np.array(f[surf_name][:])

        fig = plt.figure()
        CS1 = plt.contour(X, Y, Z, levels=np.arange(args.vmin, args.vmax, args.vlevel))
        CS2 = plt.contour(X, Y, Z, levels=np.logspace(1, 8, num=8))

        # plot trajectories
        pf = h5py.File(args.proj_file, 'r')
        plt.plot(pf['proj_xcoord'], pf['proj_ycoord'], marker='.')

        # plot red points when learning rate decays
        # for e in [150, 225, 275]:
        #     plt.plot([pf['proj_xcoord'][e]], [pf['proj_ycoord'][e]], marker='.', color='r')

        # add PCA notes
        df = h5py.File(dir_file, 'r')
        ratio_x = df['explained_variance_ratio_'][0]
        ratio_y = df['explained_variance_ratio_'][1]
        plt.xlabel('1st PC: %.2f %%' % (ratio_x * 100), fontsize='xx-large')
        plt.ylabel('2nd PC: %.2f %%' % (ratio_y * 100), fontsize='xx-large')
        df.close()
        plt.clabel(CS1, inline=1, fontsize=6)
        plt.clabel(CS2, inline=1, fontsize=6)
        fig.savefig(args.proj_file + '_' + surf_name + '_2dcontour_proj.pdf', dpi=300,
                    bbox_inches='tight', format='pdf')
        pf.close()