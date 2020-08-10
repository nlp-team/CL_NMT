import os
from fairseq import bleu, checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter
import torch
import argparse
from fairseq.data import dictionary
import subprocess


def gMain(args, fileName):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'

    utils.import_user_module(args)

    f = open(fileName, 'w')
    f.truncate()
    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    print(args)
    #     f.write(args+'\r\n')

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    # Load ensemble
    #     print('| loading model(s) from {}'.format(args.path))
    f.write('| loading model(s) from {}'.format(args.path) + '\r\n')
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    # Initialize generator
    gen_timer = StopwatchMeter()
    generator = task.build_generator(args)

    # Generate and compute BLEU score
    if args.sacrebleu:
        scorer = bleu.SacrebleuScorer()
    else:
        scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
    num_sentences = 0
    has_target = True
    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()
        for sample in t:
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if 'net_input' not in sample:
                continue

            prefix_tokens = None
            if args.prefix_size > 0:
                prefix_tokens = sample['target'][:, :args.prefix_size]

            gen_timer.start()
            hypos = task.inference_step(generator, models, sample, prefix_tokens)
            num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
            gen_timer.stop(num_generated_tokens)

            for i, sample_id in enumerate(sample['id'].tolist()):
                has_target = sample['target'] is not None

                # Remove padding
                src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad())
                target_tokens = None
                if has_target:
                    target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()

                # Either retrieve the original sentences or regenerate them from tokens.
                if align_dict is not None:
                    src_str = task.dataset(args.gen_subset).src.get_original_text(sample_id)
                    target_str = task.dataset(args.gen_subset).tgt.get_original_text(sample_id)
                else:
                    if src_dict is not None:
                        src_str = src_dict.string(src_tokens, args.remove_bpe)
                    else:
                        src_str = ""
                    if has_target:
                        target_str = tgt_dict.string(target_tokens, args.remove_bpe, escape_unk=True)

                if not args.quiet:
                    if src_dict is not None:
                        #                         print('S-{}\t{}'.format(sample_id, src_str))
                        f.write('S-{}\t{}'.format(sample_id, src_str) + '\r\n')
                    if has_target:
                        #                         print('T-{}\t{}'.format(sample_id, target_str))
                        f.write('T-{}\t{}'.format(sample_id, target_str) + '\r\n')

                # Process top predictions
                for i, hypo in enumerate(hypos[i][:min(len(hypos), args.nbest)]):
                    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypo['tokens'].int().cpu(),
                        src_str=src_str,
                        alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                        align_dict=align_dict,
                        tgt_dict=tgt_dict,
                        remove_bpe=args.remove_bpe,
                    )

                    if not args.quiet:
                        #                         print('H-{}\t{}\t{}'.format(sample_id, hypo['score'], hypo_str))
                        f.write('H-{}\t{}\t{}'.format(sample_id, hypo['score'], hypo_str) + '\r\n')
                        #                         print('P-{}\t{}'.format(
                        #                             sample_id,
                        #                             ' '.join(map(
                        #                                 lambda x: '{:.4f}'.format(x),
                        #                                 hypo['positional_scores'].tolist(),
                        #                             ))
                        #                         ))
                        f.write('P-{}\t{}'.format(
                            sample_id,
                            ' '.join(map(
                                lambda x: '{:.4f}'.format(x),
                                hypo['positional_scores'].tolist(),
                            ))
                        ) + '\r\n')
                        if args.print_alignment:
                            #                             print('A-{}\t{}'.format(
                            #                                 sample_id,
                            #                                 ' '.join(map(lambda x: str(utils.item(x)), alignment))
                            #                             ))
                            f.write('A-{}\t{}'.format(
                                sample_id,
                                ' '.join(map(lambda x: str(utils.item(x)), alignment))
                            ) + '\r\n')

                    # Score only the top hypothesis
                    if has_target and i == 0:
                        if align_dict is not None or args.remove_bpe is not None:
                            # Convert back to tokens for evaluation with unk replacement and/or without BPE
                            target_tokens = tgt_dict.encode_line(target_str, add_if_not_exist=True)
                        if hasattr(scorer, 'add_string'):
                            scorer.add_string(target_str, hypo_str)
                        else:
                            scorer.add(target_tokens, hypo_tokens)

            wps_meter.update(num_generated_tokens)
            t.log({'wps': round(wps_meter.avg)})
            num_sentences += sample['nsentences']

    #     print('| Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
    #         num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))
    f.write('| Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg) + '\r\n')
    if has_target:
        #         print('| Generate {} with beam={}: {}'.format(args.gen_subset, args.beam, scorer.result_string()))
        f.write('| Generate {} with beam={}: {}'.format(args.gen_subset, args.beam, scorer.result_string()) + '\r\n')
    f.close()
    return scorer


class args:
    beam = 5
    cpu = False
    criterion = 'cross_entropy'
    data = 'data-bin/iwslt14.tokenized.de-en/'
    dataset_impl = 'cached'
    diverse_beam_groups = -1
    diverse_beam_strength = 0.5
    force_anneal = None
    fp16 = False
    fp16_init_scale = 128
    fp16_scale_tolerance = 0.0
    fp16_scale_window = None
    gen_subset = 'test'
    lazy_load = False
    left_pad_source = 'True'
    left_pad_target = 'False'
    lenpen = 1  # 默认 1
    log_format = None
    log_interval = 1000
    lr_scheduler = 'fixed'
    lr_shrink = 0.1
    match_source_len = False
    max_len_a = 0
    max_len_b = 200
    max_sentences = 128
    max_source_positions = 1024
    max_target_positions = 1024
    max_tokens = None
    memory_efficient_fp16 = False
    min_len = 1
    min_loss_scale = 0.0001
    model_overrides = '{}'
    momentum = 0.99
    nbest = 1
    no_beamable_mm = False
    no_early_stop = False
    no_progress_bar = False
    no_repeat_ngram_size = 0
    num_shards = 1
    num_workers = 0
    optimizer = 'nag'
    path = 'checkpoints/transformer-iwslt14-de-en-sgd/checkpoint_best.pt'
    prefix_size = 0
    print_alignment = False
    quiet = False
    raw_text = False
    remove_bpe = '@@ '
    replace_unk = None
    required_batch_size_multiple = 8
    results_path = None
    sacrebleu = False
    sampling = False
    sampling_topk = -1
    score_reference = False
    seed = 1
    shard_id = 0
    skip_invalid_size_inputs_valid_test = False
    source_lang = None
    target_lang = None
    task = 'translation'
    tbmf_wrapper = False
    temperature = 1.0
    tensorboard_logdir = ''
    threshold_loss_scale = None
    unkpen = 0
    unnormalized = False
    upsample_primary = 1
    user_dir = None
    warmup_updates = 0
    weight_decay = 0.0
    batch_size = 128


class args_score:
    # sys = 'generate.sys'
    # ref = 'generate.ref'
    o = 4
    order = 4
    ignore_case = False
    sacrebleu = False
    sentence_bleu = False


def readlines(fd):
    for line in fd.readlines():
        if args_score.ignore_case:
            yield line.lower()
        else:
            yield line


def score(fdsys, tofile,refFile):
    dict = dictionary.Dictionary()
    with open(refFile) as fdref:
        scorer = bleu.Scorer(dict.pad(), dict.eos(), dict.unk())
        for sys_tok, ref_tok in zip(readlines(fdsys), readlines(fdref)):
            sys_tok = dict.encode_line(sys_tok)
            ref_tok = dict.encode_line(ref_tok)
            scorer.add(ref_tok, sys_tok)
        print(scorer.result_string(args_score.order))
        with open(tofile, 'a') as f:
            f.write(scorer.result_string(args_score.order) + '\r\n')


def get_checkpoints_bleu(dataset='iwslt14.tokenized.de-en', prefix='iwslt14-de-en', optimizer='sgd', startepoch=1, maxepoch=50,
                         savedepoch=1):
    root = '../checkpoint/' + prefix + '-' + optimizer.lower()
    files = []
    tmp = '../curve_bleu/'+dataset+'_'
    tofile = '../curve_bleu/' + dataset
    if os.path.isdir(tofile):
        pass
    else:
        os.mkdir(tofile)
    tofile += '/' + optimizer + '.txt'
    for i in range(startepoch, maxepoch + savedepoch, savedepoch):
        files.append(root + '/checkpoint' + str(i) + '.pt')

    for file in files:
        #         parser = options.get_generation_parser()
        #         args = options.parse_args_and_arch(parser)
        args.data = 'data-bin/'+dataset
        args.path = file
        args.remove_bpe = '@@ '
        args.beam = 4
        args.batch_size = 128
        args.max_len_a = 0
        args.max_len_b = 200
        gMain(args, tmp+'generate_'+optimizer+'.out')

        process = subprocess.Popen("grep ^T "+tmp+"generate_"+optimizer+".out | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > "+tmp+"generate_"+optimizer+".ref", shell= True, stdout = subprocess.PIPE)
        process.wait()
        process = subprocess.Popen("grep ^H "+tmp+"generate_"+optimizer+".out |cut -f3- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > "+tmp+"generate_"+optimizer+".sys", shell = True, stdout = subprocess.PIPE)
        process.wait()
        with open(tmp+"generate_"+optimizer+".sys", 'r') as f:
            score(f, tofile,tmp+"generate_"+optimizer+".ref")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate the bleu ...')
    parser.add_argument('--dataset', default='iwslt14-de-en', help='dataset')
    parser.add_argument('--prefix', default='iwlst14-de-en', help='dataset')
    parser.add_argument('--optimizer',default='sgd',help='The optimizer')
    parser.add_argument('--start_epoch', default=1, type=int, help='min index of epochs')
    parser.add_argument('--max_epoch', default=50, type=int, help='max number of epochs')
    parser.add_argument('--save_epoch', default=1, type=int, help='save models every few epochs')

    args_input = parser.parse_args()

    get_checkpoints_bleu(dataset=args_input.dataset, prefix=args_input.prefix, optimizer=args_input.optimizer, startepoch=args_input.start_epoch,
                         maxepoch=args_input.max_epoch,savedepoch=args_input.save_epoch)