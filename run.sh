python3 train.py --arch 'mlp' --dataset_type 'moon' --eps 0.0005 --gamma 0.1 --loss_type 'CE' --lr 0.01 --max_epoch 10000 --min_size 50 --num_classes 2 --root_log 'log' --root_model 'checkpoint' --seed 3 --store_name 'moon' --train_rule 'None'

wait

python3 test.py --arch 'mlp' --dataset_type 'moon' --eps 0.0005 --gamma 0.1 --loss_type 'CE' --lr 0.01 --max_epoch 10000 --min_size 50 --num_classes 2 --root_log 'log' --root_model 'checkpoint' --seed 3 --store_name 'moon' --train_rule 'None'
