echo "\n####### running time is $(date) #######\n" >> ./logs/UnIMP_chunksize.txt
datasets=("power_consumption")
chunk_sizes=(128 256 512 1024 2048)
for dataset in "${datasets[@]}"
do
    for chunk_size in "${chunk_sizes[@]}"
    do
        CUDA_VISIBLE_DEVICES=3 python3 -u main.py --epochs 100 --header_type Linear --chunk_size $chunk_size --chunk_batch 64 --device 0 --eval_epoch_gap 500 --mode finetune --data $dataset --load_model_name Linear_Epoch3999_allinone_v3 >> ./logs/UnIMP_chunksize.txt 2>&1
        echo "Finished processing dataset: $dataset, $chunk_size" >> ./logs/UnIMP_chunksize.txt
    done
done &