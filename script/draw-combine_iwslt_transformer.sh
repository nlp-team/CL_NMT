dataset=iwslt14-de-en
checkpath=checkpoint

# adam_inv_1e-3
python3 plot_2D.py --dir_file=${checkpath}/${dataset}-adam_inv_1e-3/PCA_lr=0.001_optimier=adam_ignore_embedding=False_ignoreBN/directions.h5 \
--proj_file=${checkpath}/${dataset}-adam_inv_1e-3/PCA_lr=0.001_optimier=adam_ignore_embedding=False_ignoreBN/directions.h5_proj_cos.h5 \
--surf_file=${checkpath}/${dataset}-adam_inv_1e-3/PCA_lr=0.001_optimier=adam_ignore_embedding=False_ignoreBN/directions.h5_surf_[-10.0,300.0,20]x[-80.0,60.0,20].h5

# adam_inv_5e-4
python3 plot_2D.py --dir_file=${checkpath}/${dataset}-adam_inv_5e-4/PCA_lr=0.0005_optimier=adam_ignore_embedding=False_ignoreBN/directions.h5 \
--proj_file=${checkpath}/${dataset}-adam_inv_5e-4/PCA_lr=0.0005_optimier=adam_ignore_embedding=False_ignoreBN/directions.h5_proj_cos.h5 \
--surf_file=${checkpath}/${dataset}-adam_inv_5e-4/PCA_lr=0.0005_optimier=adam_ignore_embedding=False_ignoreBN/directions.h5_surf_[-10.0,175.0,20]x[-50.0,60.0,20].h5

# adam_cyc_yshrink_5e-4
python3 plot_2D.py --dir_file=${checkpath}/${dataset}-adam_cyc_yshrink_5e-4/PCA_lr=1e-05_optimier=adam_ignore_embedding=False_ignoreBN/directions.h5 \
--proj_file=${checkpath}/${dataset}-adam_cyc_yshrink_5e-4/PCA_lr=1e-05_optimier=adam_ignore_embedding=False_ignoreBN/directions.h5_proj_cos.h5 \
--surf_file=${checkpath}/${dataset}-adam_cyc_yshrink_5e-4/PCA_lr=1e-05_optimier=adam_ignore_embedding=False_ignoreBN/directions.h5_surf_[-10.0,170.0,20]x[-50.0,70.0,20].h5




