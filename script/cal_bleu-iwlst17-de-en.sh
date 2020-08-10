cd fairseq_master

dataset=iwslt17.tokenized.de-en
prefix=iwslt17-de-en

# adam_cyc_nshrink_7.6e-4
CUDA_VISIBLE_DEVICES=6 python3 cal_bleu_curve.py --dataset=$dataset --prefix=$prefix --optimizer=adam_cyc_nshrink_7.6e-4 --start_epoch=1 --max_epoch=50 --save_epoch=1 

# adam_cyc_yshrink_7.6e-4
CUDA_VISIBLE_DEVICES=6 python3 cal_bleu_curve.py --dataset=$dataset --prefix=$prefix --optimizer=adam_cyc_yshrink_7.6e-4 --start_epoch=1 --max_epoch=50 --save_epoch=1 

# adam_inv_7.6e-4
CUDA_VISIBLE_DEVICES=6 python3 cal_bleu_curve.py --dataset=$dataset --prefix=$prefix --optimizer=adam_inv_7.6e-4 --start_epoch=1 --max_epoch=50 --save_epoch=1 

# adam_inv_5e-4
CUDA_VISIBLE_DEVICES=6 python3 cal_bleu_curve.py --dataset=$dataset --prefix=$prefix --optimizer=adam_inv_5e-4 --start_epoch=1 --max_epoch=50 --save_epoch=1

# adam_inv_3e-4
CUDA_VISIBLE_DEVICES=6 python3 cal_bleu_curve.py --dataset=$dataset --prefix=$prefix --optimizer=adam_inv_3e-4 --start_epoch=1 --max_epoch=50 --save_epoch=1

# sgd_cyc_nshrink_8
CUDA_VISIBLE_DEVICES=6 python3 cal_bleu_curve.py --dataset=$dataset --prefix=$prefix --optimizer=sgd_cyc_nshrink_8 --start_epoch=1 --max_epoch=50 --save_epoch=1 

# sgd_inv_30
CUDA_VISIBLE_DEVICES=6 python3 cal_bleu_curve.py --dataset=$dataset --prefix=$prefix --optimizer=sgd_inv_30 --start_epoch=1 --max_epoch=50 --save_epoch=1 

