cd fairseq_master

datapath=data-bin/iwslt14.tokenized.de-en
src=de
dataset=iwslt14-de-en
checkpath=../checkpoint
tbpath=../tensorboardLog
logpath=../train_log

# adam_lr_range_test
lrs=adam_lr_range_test
CUDA_VISIBLE_DEVICES=7 python3 train.py $datapath \
--arch transformer_iwslt_de_en  \
--source-lang $src --target-lang en \
--optimizer adam --clip-norm 0.1 \
--lr 0.00001 \
--lr-scheduler adam_range_test \
--max-tokens  4096 \
--save-dir ${checkpath}/${dataset}-$lrs \
--max-epoch 35 \
--tensorboard-logdir ${tbpath}/${dataset}-$lrs \
--no-progress-bar --log-interval 50 \
--keep-last-epochs 10 > $logpath/log_${dataset}_$lrs.out 


