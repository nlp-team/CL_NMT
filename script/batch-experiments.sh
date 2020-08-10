#! /bin/bash -v

cd ../fairseq_master

src=de
tgt=en
data=iwslt14
GPU=0

pair=${src}-${tgt}
datapath=data-bin/${data}.tokenized.${pair}
dataset=${data}-${pair}
checkpath=../checkpoint
tbpath=../tensorboardLog
logpath=../train_log


train_model(){
    optimizer=$1
    lr_schedule=$2
    max_lr=$3
    min_lr=$4
    batch=$5
    shrink=$6
    extra_args=$7
    suffix=$8
    case $batch in
         256) period=91000 ;;
        1024) period=22290 ;;
        4096) period=5000 ;;
    esac
    case $shrink in
         1) shrink_w="nshrink" ;;
         *) shrink_w="${shrink}-shrink" ;;
    esac
    lrs=${optimizer}_${lr_schedule}_${shrink_w}_${max_lr}_${batch}${suffix}
    CUDA_VISIBLE_DEVICES=$GPU python3 train.py $datapath \
    --arch transformer_iwslt_de_en  \
    --source-lang $src --target-lang $tgt \
    --optimizer ${optimizer} --clip-norm 0.1 \
    --lr ${min_lr} \
    --lr-scheduler ${lr_schedule} ${extra_args} \
    --max-tokens  ${batch} \
    --save-dir ${checkpath}/${dataset}-$lrs \
    --max-epoch 50 \
    --tensorboard-logdir ${tbpath}/${dataset}-$lrs \
    --no-progress-bar --log-interval 50 \
    --max-lr ${max_lr} --lr-period-updates ${period} --lr-shrink ${shrink} > $logpath/log_${dataset}_${lrs}.out
}

train_inverse(){
    optimizer=$1
    lr_schedule=$2
    max_lr=$3
    min_lr=$4
    batch=$5
    shrink=$6
    extra_args=$7
    suffix=$8
    case $batch in
         256) period=91000 ;;
        1024) period=22290 ;;
        4096) period=5000 ;;
    esac
    case $shrink in
         1) shrink_w="nshrink" ;;
         *) shrink_w="${shrink}-shrink" ;;
    esac
    lrs=${optimizer}_${lr_schedule}_${max_lr}_${batch}${suffix}
    CUDA_VISIBLE_DEVICES=$GPU python3 train.py $datapath \
    --arch transformer_iwslt_de_en  \
    --source-lang $src --target-lang $tgt \
    --optimizer ${optimizer} --clip-norm 0.1 \
    --lr ${max_lr} \
    --lr-scheduler ${lr_schedule} ${extra_args} \
    --max-tokens  ${batch} \
    --save-dir ${checkpath}/${dataset}-$lrs \
    --max-epoch 50 \
    --tensorboard-logdir ${tbpath}/${dataset}-$lrs \
    --no-progress-bar --log-interval 50 > $logpath/log_${dataset}_$lrs.out
}

# max_lr=5e-4
# lr_schedule=sine
train_model "adam" "sine" "5e-4" "1e-5" "4096" "1" "" ""
train_model "adam" "sine" "5e-4" "1e-5" "1024" "1" "" ""
train_model "adam" "sine" "5e-4" "1e-5" "256" "1" "" ""

# max_lr=5e-4
# lr_schedule=full_cosine
train_model "adam" "full_cosine" "5e-4" "1e-5" "4096" "1" "--warmup-init-lr 1e-07 --warmup-updates 400" "_with-warmup"
train_model "adam" "full_cosine" "5e-4" "1e-5" "1024" "1" "--warmup-init-lr 1e-07 --warmup-updates 1600" "_with-warmup"
train_model "adam" "full_cosine" "5e-4" "1e-5" "256" "1" "--warmup-init-lr 1e-07 --warmup-updates 6400" "_with-warmup"

# max_lr=5e-4
# lr_schedule=triangular
train_model "adam" "triangular" "5e-4" "1e-5" "4096" "1" "" ""
train_model "adam" "triangular" "5e-4" "1e-5" "1024" "1" "" ""
train_model "adam" "triangular" "5e-4" "1e-5" "256" "1" "" ""

# max_lr=5e-4
# lr_schedule=inverse_sqrt
train_inverse "adam" "inverse_sqrt" "5e-4" "1e-5" "4096" "1" "--warmup-init-lr 1e-07 --warmup-updates 400" ""
train_inverse "adam" "inverse_sqrt" "5e-4" "1e-5" "1024" "1" "--warmup-init-lr 1e-07 --warmup-updates 1600" ""
train_inverse "adam" "inverse_sqrt" "5e-4" "1e-5" "256" "1" "--warmup-init-lr 1e-07 --warmup-updates 6400" ""

# -------------------------------------------------------------

# max_lr=3e-4
# lr_schedule=sine
train_model "adam" "sine" "3e-4" "1e-5" "4096" "1" "" ""
train_model "adam" "sine" "3e-4" "1e-5" "1024" "1" "" ""
train_model "adam" "sine" "3e-4" "1e-5" "256" "1" "" ""

# max_lr=3e-4
# lr_schedule=full_cosine
train_model "adam" "full_cosine" "3e-4" "1e-5" "4096" "1" "--warmup-init-lr 1e-07 --warmup-updates 400" "_with-warmup"
train_model "adam" "full_cosine" "3e-4" "1e-5" "1024" "1" "--warmup-init-lr 1e-07 --warmup-updates 1600" "_with-warmup"
train_model "adam" "full_cosine" "3e-4" "1e-5" "256" "1" "--warmup-init-lr 1e-07 --warmup-updates 6400" "_with-warmup"

# max_lr=3e-4
# lr_schedule=triangular
train_model "adam" "triangular" "3e-4" "1e-5" "4096" "1" "" ""
train_model "adam" "triangular" "3e-4" "1e-5" "1024" "1" "" ""
train_model "adam" "triangular" "3e-4" "1e-5" "256" "1" "" ""

# max_lr=3e-4
# lr_schedule=inverse_sqrt
train_inverse "adam" "inverse_sqrt" "3e-4" "1e-5" "4096" "1" "--warmup-init-lr 1e-07 --warmup-updates 400" ""
train_inverse "adam" "inverse_sqrt" "3e-4" "1e-5" "1024" "1" "--warmup-init-lr 1e-07 --warmup-updates 1600" ""
train_inverse "adam" "inverse_sqrt" "3e-4" "1e-5" "256" "1" "--warmup-init-lr 1e-07 --warmup-updates 6400" ""



