echo "\n####### running time is $(date) #######\n" >> ./logs/UnIMP_batchsize.txt
datasets=("power_consumption")
batch_sizes=(16 32 64 128 256)
for dataset in "${datasets[@]}"
do
    for batch_size in "${batch_sizes[@]}"
    do
        CUDA_VISIBLE_DEVICES=7 python3 -u main.py --epochs 100 --header_type Linear --chunk_size 512 --chunk_batch $batch_size --device 0 --eval_epoch_gap 100 --mode finetune --data $dataset --load_model_name Linear_Epoch3999_allinone_v3 >> ./logs/UnIMP_batchsize.txt 2>&1
        echo "Finished processing dataset: $dataset, $batch_size" >> ./logs/UnIMP_batchsize.txt
    done
done &