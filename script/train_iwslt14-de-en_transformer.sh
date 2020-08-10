cd fairseq_master

datapath=data-bin/iwslt14.tokenized.de-en
src=de
dataset=iwslt14-de-en
checkpath=../checkpoint
tbpath=../tensorboardLog
logpath=../train_log

# adam_cyc_nshrink_5e-4
lrs=adam_cyc_nshrink_5e-4
CUDA_VISIBLE_DEVICES=0 python3 train.py $datapath \
--arch transformer_iwslt_de_en  \
--source-lang $src --target-lang en \
--optimizer adam --clip-norm 0.1 \
--lr 0.00001 \
--lr-scheduler triangular \
--max-tokens  4096 \
--save-dir ${checkpath}/${dataset}-$lrs \
--max-epoch 50 \
--tensorboard-logdir ${tbpath}/${dataset}-$lrs \
--no-progress-bar --log-interval 50 \
--max-lr 0.0005 --lr-period-updates 5000 --lr-shrink 1 > $logpath/log_${dataset}_$lrs.out 

# adam_cyc_yshrink_5e-4
lrs=adam_cyc_yshrink_5e-4
CUDA_VISIBLE_DEVICES=0 python3 train.py $datapath \
--arch transformer_iwslt_de_en  \
--source-lang $src --target-lang en \
--optimizer adam --clip-norm 0.1 \
--lr 0.00001 \
--lr-scheduler triangular \
--max-tokens  4096 \
--save-dir ${checkpath}/${dataset}-$lrs \
--max-epoch 50 \
--tensorboard-logdir ${tbpath}/${dataset}-$lrs \
--no-progress-bar --log-interval 50 \
--max-lr 0.0005 --lr-period-updates 5000 --lr-shrink 0.5 > $logpath/log_${dataset}_$lrs.out 

# adam_cyc_nshrink_1.6e-3
lrs=adam_cyc_nshrink_1.6e-3
CUDA_VISIBLE_DEVICES=0 python3 train.py $datapath \
--arch transformer_iwslt_de_en  \
--source-lang $src --target-lang en \
--optimizer adam --clip-norm 0.1 \
--lr 0.00001 \
--lr-scheduler triangular \
--max-tokens  4096 \
--save-dir ${checkpath}/${dataset}-$lrs \
--max-epoch 50 \
--tensorboard-logdir ${tbpath}/${dataset}-$lrs \
--no-progress-bar --log-interval 50 \
--max-lr 0.00166 --lr-period-updates 5000 --lr-shrink 1 > $logpath/log_${dataset}_$lrs.out 

# adam_cyc_yshrink_1.6e-3
lrs=adam_cyc_yshrink_1.6e-3
CUDA_VISIBLE_DEVICES=0 python3 train.py $datapath \
--arch transformer_iwslt_de_en  \
--source-lang $src --target-lang en \
--optimizer adam --clip-norm 0.1 \
--lr 0.00001 \
--lr-scheduler triangular \
--max-tokens  4096 \
--save-dir ${checkpath}/${dataset}-$lrs \
--max-epoch 50 \
--tensorboard-logdir ${tbpath}/${dataset}-$lrs \
--no-progress-bar --log-interval 50 \
--max-lr 0.00166 --lr-period-updates 5000 --lr-shrink 0.5 > $logpath/log_${dataset}_$lrs.out 

# adam_inv_1e-3
lrs=adam_inv_1e-3
CUDA_VISIBLE_DEVICES=0 python3 train.py $datapath \
--arch transformer_iwslt_de_en  \
--source-lang $src --target-lang en \
--optimizer adam --clip-norm 0.1 \
--lr 0.001 \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 400 \
--max-tokens  4096 \
--save-dir ${checkpath}/${dataset}-$lrs \
--max-epoch 50 \
--tensorboard-logdir ${tbpath}/${dataset}-$lrs \
--no-progress-bar --log-interval 50 > $logpath/log_${dataset}_$lrs.out 

# adam_inv_5e-4
lrs=adam_inv_5e-4
CUDA_VISIBLE_DEVICES=0 python3 train.py $datapath \
--arch transformer_iwslt_de_en  \
--source-lang $src --target-lang en \
--optimizer adam --clip-norm 0.1 \
--lr 0.0005 \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 400 \
--max-tokens  4096 \
--save-dir ${checkpath}/${dataset}-$lrs \
--max-epoch 50 \
--tensorboard-logdir ${tbpath}/${dataset}-$lrs \
--no-progress-bar --log-interval 50 > $logpath/log_${dataset}_$lrs.out 

# adam_inv_3e-4
lrs=adam_inv_3e-4
CUDA_VISIBLE_DEVICES=0 python3 train.py $datapath \
--arch transformer_iwslt_de_en  \
--source-lang $src --target-lang en \
--optimizer adam --clip-norm 0.1 \
--lr 0.0003 \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 400 \
--max-tokens  4096 \
--save-dir ${checkpath}/${dataset}-$lrs \
--max-epoch 50 \
--tensorboard-logdir ${tbpath}/${dataset}-$lrs \
--no-progress-bar --log-interval 50 > $logpath/log_${dataset}_$lrs.out 

# adam_inv_1e-5
lrs=adam_inv_1e-5
CUDA_VISIBLE_DEVICES=0 python3 train.py $datapath \
--arch transformer_iwslt_de_en  \
--source-lang $src --target-lang en \
--optimizer adam --clip-norm 0.1 \
--lr 0.00001 \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 400 \
--max-tokens  4096 \
--save-dir ${checkpath}/${dataset}-$lrs \
--max-epoch 50 \
--tensorboard-logdir ${tbpath}/${dataset}-$lrs \
--no-progress-bar --log-interval 50 > $logpath/log_${dataset}_$lrs.out 

# sgd_cyc_nshrink_6.9
lrs=sgd_cyc_nshrink_6.9
CUDA_VISIBLE_DEVICES=0 python3 train.py $datapath \
--arch transformer_iwslt_de_en  \
--source-lang $src --target-lang en \
--optimizer sgd --clip-norm 0.1 \
--lr 0.001 \
--lr-scheduler triangular \
--max-tokens  4096 \
--save-dir ${checkpath}/${dataset}-$lrs \
--max-epoch 50 \
--tensorboard-logdir ${tbpath}/${dataset}-$lrs \
--no-progress-bar --log-interval 50 \
--max-lr 6.9 --lr-period-updates 5000 --lr-shrink 1 > $logpath/log_${dataset}_$lrs.out

# sgd_cyc_yshrink_6.9
lrs=sgd_cyc_yshrink_6.9
CUDA_VISIBLE_DEVICES=0 python3 train.py $datapath \
--arch transformer_iwslt_de_en  \
--source-lang $src --target-lang en \
--optimizer sgd --clip-norm 0.1 \
--lr 0.001 \
--lr-scheduler triangular \
--max-tokens  4096 \
--save-dir ${checkpath}/${dataset}-$lrs \
--max-epoch 50 \
--tensorboard-logdir ${tbpath}/${dataset}-$lrs \
--no-progress-bar --log-interval 50 \
--max-lr 6.9 --lr-period-updates 5000 --lr-shrink 0.5 > $logpath/log_${dataset}_$lrs.out 

# sgd_cyc_nshrink_23.2
lrs=sgd_cyc_nshrink_23.2
CUDA_VISIBLE_DEVICES=4 python3 train.py $datapath \
--arch transformer_iwslt_de_en  \
--source-lang $src --target-lang en \
--optimizer sgd --clip-norm 0.1 \
--lr 0.001 \
--lr-scheduler triangular \
--max-tokens  4096 \
--save-dir ${checkpath}/${dataset}-$lrs \
--max-epoch 50 \
--tensorboard-logdir ${tbpath}/${dataset}-$lrs \
--no-progress-bar --log-interval 50 \
--max-lr 23.2 --lr-period-updates 5000 --lr-shrink 1 > $logpath/log_${dataset}_$lrs.out

# sgd_cyc_yshrink_23.2
lrs=sgd_cyc_yshrink_23.2
CUDA_VISIBLE_DEVICES=5 python3 train.py $datapath \
--arch transformer_iwslt_de_en  \
--source-lang $src --target-lang en \
--optimizer sgd --clip-norm 0.1 \
--lr 0.001 \
--lr-scheduler triangular \
--max-tokens  4096 \
--save-dir ${checkpath}/${dataset}-$lrs \
--max-epoch 50 \
--tensorboard-logdir ${tbpath}/${dataset}-$lrs \
--no-progress-bar --log-interval 50 \
--max-lr 23.2 --lr-period-updates 5000 --lr-shrink 0.5 > $logpath/log_${dataset}_$lrs.out 

# sgd_inv_0.1
lrs=sgd_inv_0.1
CUDA_VISIBLE_DEVICES=0 python3 train.py $datapath \
--arch transformer_iwslt_de_en  \
--source-lang $src --target-lang en \
--optimizer sgd --clip-norm 0.1 \
--lr 0.1 \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 400 \
--max-tokens  4096 \
--save-dir ${checkpath}/${dataset}-$lrs \
--max-epoch 50 \
--tensorboard-logdir ${tbpath}/${dataset}-$lrs \
--no-progress-bar --log-interval 50 > $logpath/log_${dataset}_$lrs.out 

# sgd_inv_1
lrs=sgd_inv_1
CUDA_VISIBLE_DEVICES=0 python3 train.py $datapath \
--arch transformer_iwslt_de_en  \
--source-lang $src --target-lang en \
--optimizer sgd --clip-norm 0.1 \
--lr 1 \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 400 \
--max-tokens  4096 \
--save-dir ${checkpath}/${dataset}-$lrs \
--max-epoch 50 \
--tensorboard-logdir ${tbpath}/${dataset}-$lrs \
--no-progress-bar --log-interval 50 > $logpath/log_${dataset}_$lrs.out 

# sgd_inv_10
lrs=sgd_inv_10
CUDA_VISIBLE_DEVICES=0 python3 train.py $datapath \
--arch transformer_iwslt_de_en  \
--source-lang $src --target-lang en \
--optimizer sgd --clip-norm 0.1 \
--lr 10 \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 400 \
--max-tokens  4096 \
--save-dir ${checkpath}/${dataset}-$lrs \
--max-epoch 50 \
--tensorboard-logdir ${tbpath}/${dataset}-$lrs \
--no-progress-bar --log-interval 50 > $logpath/log_${dataset}_$lrs.out 

# sgd_inv_20
lrs=sgd_inv_20
CUDA_VISIBLE_DEVICES=0 python3 train.py $datapath \
--arch transformer_iwslt_de_en  \
--source-lang $src --target-lang en \
--optimizer sgd --clip-norm 0.1 \
--lr 20 \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 400 \
--max-tokens  4096 \
--save-dir ${checkpath}/${dataset}-$lrs \
--max-epoch 50 \
--tensorboard-logdir ${tbpath}/${dataset}-$lrs \
--no-progress-bar --log-interval 50 > $logpath/log_${dataset}_$lrs.out 

# sgd_inv_30
lrs=sgd_inv_30
CUDA_VISIBLE_DEVICES=0 python3 train.py $datapath \
--arch transformer_iwslt_de_en  \
--source-lang $src --target-lang en \
--optimizer sgd --clip-norm 0.1 \
--lr 30 \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 400 \
--max-tokens  4096 \
--save-dir ${checkpath}/${dataset}-$lrs \
--max-epoch 50 \
--tensorboard-logdir ${tbpath}/${dataset}-$lrs \
--no-progress-bar --log-interval 50 > $logpath/log_${dataset}_$lrs.out 

################ batch size comparison test ################
# adam_cyc_noshrink_5e-4_4096 (same as adam_cyc_nshrink_5e-4)

# adam_cyc_nshrink_5e-4_1024
lrs=adam_cyc_nshrink_5e-4_1024
CUDA_VISIBLE_DEVICES=0 python3 train.py $datapath \
--arch transformer_iwslt_de_en  \
--source-lang $src --target-lang en \
--optimizer adam --clip-norm 0.1 \
--lr 0.00001 \
--lr-scheduler triangular \
--max-tokens  1024 \
--save-dir ${checkpath}/${dataset}-$lrs \
--max-epoch 50 \
--tensorboard-logdir ${tbpath}/${dataset}-$lrs \
--no-progress-bar --log-interval 50 \
--max-lr 0.0005 --lr-period-updates 22290 --lr-shrink 1 > $logpath/log_${dataset}_$lrs.out 

# adam_cyc_nshrink_5e-4_256
lrs=adam_cyc_nshrink_5e-4_256
CUDA_VISIBLE_DEVICES=0 python3 train.py $datapath \
--arch transformer_iwslt_de_en  \
--source-lang $src --target-lang en \
--optimizer adam --clip-norm 0.1 \
--lr 0.00001 \
--lr-scheduler triangular \
--max-tokens  256 \
--save-dir ${checkpath}/${dataset}-$lrs \
--max-epoch 50 \
--tensorboard-logdir ${tbpath}/${dataset}-$lrs \
--no-progress-bar --log-interval 50 \
--max-lr 0.0005 --lr-period-updates 91000 --lr-shrink 1 > $logpath/log_${dataset}_$lrs.out

