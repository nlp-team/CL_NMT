dataset=iwslt14-de-en
checkpath=checkpoint

# adam_inv_1e-3
CUDA_VISIBLE_DEVICES=3 python3 plot_transformer_surface.py \
--model_folder=${checkpath}/${dataset}-adam_inv_1e-3/ \
--dir_file=${checkpath}/${dataset}-adam_inv_1e-3/PCA_lr=0.001_optimier=adam_ignore_embedding=False_ignoreBN/directions.h5 --x=-10:300:20 --y=-80:60:20 --ignore=biasbn 


# adam_inv_5e-4
CUDA_VISIBLE_DEVICES=3 python3 plot_transformer_surface.py \
--model_folder=${checkpath}/${dataset}-adam_inv_5e-4/ \
--dir_file=${checkpath}/${dataset}-adam_inv_5e-4/PCA_lr=0.0005_optimier=adam_ignore_embedding=False_ignoreBN/directions.h5 --x=-10:175:20 --y=-50:60:20 --ignore=biasbn 


# adam_cyc_yshrink_5e-4
CUDA_VISIBLE_DEVICES=3 python3 plot_transformer_surface.py \
--model_folder=${checkpath}/${dataset}-adam_cyc_yshrink_5e-4/ \
--dir_file=${checkpath}/${dataset}-adam_cyc_yshrink_5e-4/PCA_lr=1e-05_optimier=adam_ignore_embedding=False_ignoreBN/directions.h5 --x=-10:170:20 --y=-50:70:20 --ignore=biasbn 

