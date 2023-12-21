## office31
python train_uda.py --pl $1 --tar_par $2 --svd_par $3 --max_epoch 100 --method $4 --dset office31 --s 0 --t 1 --output logs/uda/ --worker 3 --batch_size 32 --momentum $5 --ifsvd --laplac $6 --ap $7 

python train_uda.py --pl $1 --tar_par $2 --svd_par $3 --max_epoch 100 --method $4 --dset office31 --s 0 --t 2 --output logs/uda/ --worker 3 --batch_size 32 --momentum $5 --ifsvd --laplac $6 --ap $7

#
python train_uda.py --pl $1 --tar_par $2 --svd_par $3 --max_epoch 100 --method $4 --dset office31 --s 1 --t 0 --output logs/uda/ --worker 3 --batch_size 32 --momentum $5 --ifsvd --laplac $6 --ap $7

python train_uda.py --pl $1 --tar_par $2 --svd_par $3 --max_epoch 100 --method $4 --dset office31 --s 1 --t 2 --output logs/uda/ --worker 3 --batch_size 32 --momentum $5 --ifsvd --laplac $6 --ap $7

#
python train_uda.py --pl $1 --tar_par $2 --svd_par $3 --max_epoch 100 --method $4 --dset office31 --s 2 --t 0 --output logs/uda/ --worker 3 --batch_size 32 --momentum $5 --ifsvd --laplac $6 --ap $7

python train_uda.py --pl $1 --tar_par $2 --svd_par $3 --max_epoch 100 --method $4 --dset office31 --s 2 --t 1 --output logs/uda/ --worker 3 --batch_size 32 --momentum $5 --ifsvd --laplac $6 --ap $7


#
python train_uda.py --pl $1 --tar_par $2 --svd_par $3 --max_epoch 100 --method $4 --dset office31 --s 0 --t 1 --output logs/uda/ --worker 3 --batch_size 32 --momentum $5 --ifsvd --laplac $6 --ap $7 --ifcorrect

python train_uda.py --pl $1 --tar_par $2 --svd_par $3 --max_epoch 100 --method $4 --dset office31 --s 0 --t 2 --output logs/uda/ --worker 3 --batch_size 32 --momentum $5 --ifsvd --laplac $6 --ap $7 --ifcorrect

#
python train_uda.py --pl $1 --tar_par $2 --svd_par $3 --max_epoch 100 --method $4 --dset office31 --s 1 --t 0 --output logs/uda/ --worker 3 --batch_size 32 --momentum $5 --ifsvd --laplac $6 --ap $7 --ifcorrect

python train_uda.py --pl $1 --tar_par $2 --svd_par $3 --max_epoch 100 --method $4 --dset office31 --s 1 --t 2 --output logs/uda/ --worker 3 --batch_size 32 --momentum $5 --ifsvd --laplac $6 --ap $7 --ifcorrect

#
python train_uda.py --pl $1 --tar_par $2 --svd_par $3 --max_epoch 100 --method $4 --dset office31 --s 2 --t 0 --output logs/uda/ --worker 3 --batch_size 32 --momentum $5 --ifsvd --laplac $6 --ap $7 --ifcorrect

python train_uda.py --pl $1 --tar_par $2 --svd_par $3 --max_epoch 100 --method $4 --dset office31 --s 2 --t 1 --output logs/uda/ --worker 3 --batch_size 32 --momentum $5 --ifsvd --laplac $6 --ap $7 --ifcorrect
