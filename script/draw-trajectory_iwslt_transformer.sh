dataset=iwslt14-de-en
checkpath=checkpoint

# adam_inv_1e-3
python3 plot_transformer_trajectory.py \
--model_folder=${checkpath}/${dataset}-adam_inv_1e-3/ \
--ignore=biasbn --start_epoch=1 --max_epoch=50 --save_epoch=1 

# adam_inv_5e-4
python3 plot_transformer_trajectory.py \
--model_folder=${checkpath}/${dataset}-adam_inv_5e-4/ \
--ignore=biasbn --start_epoch=1 --max_epoch=50 --save_epoch=1 

# adam_cyc_yshrink_5e-4
python3 plot_transformer_trajectory.py \
--model_folder=${checkpath}/${dataset}-adam_cyc_yshrink_5e-4/ \
--ignore=biasbn --start_epoch=1 --max_epoch=50 --save_epoch=1 


