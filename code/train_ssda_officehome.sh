## office_home
python train_ssda.py --pl $1 --tar_par $2 --svd_par $3 --max_epoch 50 --method $4 --dset office_home --s 0 --t 1 --output logs/ssda/ --worker 3 --batch_size 32 --momentum $5 --laplac $6 --ap $7 --shot $8 

python train_ssda.py --pl $1 --tar_par $2 --svd_par $3 --max_epoch 50 --method $4 --dset office_home --s 0 --t 2 --output logs/ssda/ --worker 3 --batch_size 32 --momentum $5 --laplac $6 --ap $7 --shot $8

python train_ssda.py --pl $1 --tar_par $2 --svd_par $3 --max_epoch 50 --method $4 --dset office_home --s 0 --t 3 --output logs/ssda/ --worker 3 --batch_size 32 --momentum $5 --laplac $6 --ap $7 --shot $8

#
python train_ssda.py --pl $1 --tar_par $2 --svd_par $3 --max_epoch 50 --method $4 --dset office_home --s 1 --t 0 --output logs/ssda/ --worker 3 --batch_size 32 --momentum $5 --laplac $6 --ap $7 --shot $8

python train_ssda.py --pl $1 --tar_par $2 --svd_par $3 --max_epoch 50 --method $4 --dset office_home --s 1 --t 2 --output logs/ssda/ --worker 3 --batch_size 32 --momentum $5 --laplac $6 --ap $7 --shot $8

python train_ssda.py --pl $1 --tar_par $2 --svd_par $3 --max_epoch 50 --method $4 --dset office_home --s 1 --t 3 --output logs/ssda/ --worker 3 --batch_size 32 --momentum $5 --laplac $6 --ap $7 --shot $8

#
python train_ssda.py --pl $1 --tar_par $2 --svd_par $3 --max_epoch 50 --method $4 --dset office_home --s 2 --t 0 --output logs/ssda/ --worker 3 --batch_size 32 --momentum $5 --laplac $6 --ap $7 --shot $8

python train_ssda.py --pl $1 --tar_par $2 --svd_par $3 --max_epoch 50 --method $4 --dset office_home --s 2 --t 1 --output logs/ssda/ --worker 3 --batch_size 32 --momentum $5 --laplac $6 --ap $7 --shot $8

python train_ssda.py --pl $1 --tar_par $2 --svd_par $3 --max_epoch 50 --method $4 --dset office_home --s 2 --t 3 --output logs/ssda/ --worker 3 --batch_size 32 --momentum $5 --laplac $6 --ap $7 --shot $8

#
python train_ssda.py --pl $1 --tar_par $2 --svd_par $3 --max_epoch 50 --method $4 --dset office_home --s 3 --t 0 --output logs/ssda/ --worker 3 --batch_size 32 --momentum $5 --laplac $6 --ap $7 --shot $8

python train_ssda.py --pl $1 --tar_par $2 --svd_par $3 --max_epoch 50 --method $4 --dset office_home --s 3 --t 1 --output logs/ssda/ --worker 3 --batch_size 32 --momentum $5 --laplac $6 --ap $7 --shot $8

python train_ssda.py --pl $1 --tar_par $2 --svd_par $3 --max_epoch 50 --method $4 --dset office_home --s 3 --t 2 --output logs/ssda/ --worker 3 --batch_size 32 --momentum $5 --laplac $6 --ap $7 --shot $8
