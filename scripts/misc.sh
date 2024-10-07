CUDA_VISIBLE_DEVICES=2 nohup python3 -u main.py --epochs 20000 --header_type Linear --chunk_size 512 --chunk_batch 64 --device 0 --eval_epoch_gap 200 >> ./logs/All_Linear_v1.txt 2>&1 &


CUDA_VISIBLE_DEVICES=0 python3 -u main.py --epochs 1000 --header_type Linear --chunk_size 512 --chunk_batch 64 --device 0 --eval_epoch_gap 100 --mode finetune --data parkinsons --load_model_name Linear_Epoch3999_allinone_v3

CUDA_VISIBLE_DEVICES=0 python3 -u main.py --header_type Linear --chunk_size 512 --chunk_batch 64 --device 0 --mode testing --load_model_name Linear_Epoch2499_allinone_v1

CUDA_VISIBLE_DEVICES=0 python3 -u main.py --epochs 1000 --header_type Linear --chunk_size 512 --chunk_batch 64 --device 0 --eval_epoch_gap 100 --mode finetune --missing_mechanism MAR --data libras --load_model_name Linear_Epoch3999_allinone_v1


CUDA_VISIBLE_DEVICES=1 python3 -u main.py --epochs 1000 --header_type Linear --chunk_size 512 --chunk_batch 64 --device 0 --eval_epoch_gap 100 --mode finetune --missing_mechanism MNAR --data libras --load_model_name Linear_Epoch3999_allinone_v3


# CUDA_VISIBLE_DEVICES=2,3 python3 -u main.py --header_type LLM --chunk_size 512 --chunk_batch 64 --device 0 --mode testing --load_model_name Linear_Epoch2499_allinone_v1


