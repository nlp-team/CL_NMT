cd fairseq_master

datapath=data-bin/iwslt14.tokenized.fr-en
src=fr
dataset=iwslt14-fr-en
checkpath=../checkpoint
tbpath=../tensorboardLog
logpath=../train_log


# adam_cyc_nshrink_8e-4
lrs=adam_cyc_nshrink_8e-4
CUDA_VISIBLE_DEVICES=3 python3 train.py $datapath \
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
--max-lr 0.0008 --lr-period-updates 5000 --lr-shrink 1 > $logpath/log_${dataset}_$lrs.out 

# adam_cyc_yshrink_8e-4
lrs=adam_cyc_yshrink_8e-4
CUDA_VISIBLE_DEVICES=3 python3 train.py $datapath \
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
--max-lr 0.0008 --lr-period-updates 5000 --lr-shrink 0.5 > $logpath/log_${dataset}_$lrs.out 

# adam_inv_1e-3
lrs=adam_inv_1e-3
CUDA_VISIBLE_DEVICES=3 python3 train.py $datapath \
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
CUDA_VISIBLE_DEVICES=3 python3 train.py $datapath \
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
CUDA_VISIBLE_DEVICES=3 python3 train.py $datapath \
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

# adam_inv_8e-4
lrs=adam_inv_8e-4
CUDA_VISIBLE_DEVICES=3 python3 train.py $datapath \
--arch transformer_iwslt_de_en  \
--source-lang $src --target-lang en \
--optimizer adam --clip-norm 0.1 \
--lr 0.0008 \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 400 \
--max-tokens  4096 \
--save-dir ${checkpath}/${dataset}-$lrs \
--max-epoch 50 \
--tensorboard-logdir ${tbpath}/${dataset}-$lrs \
--no-progress-bar --log-interval 50 > $logpath/log_${dataset}_$lrs.out 

# adam_inv_1e-5
lrs=adam_inv_1e-5
CUDA_VISIBLE_DEVICES=3 python3 train.py $datapath \
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

# sgd_inv_30
lrs=sgd_inv_30
CUDA_VISIBLE_DEVICES=3 python3 train.py $datapath \
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