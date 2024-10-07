echo "\n####### running time is $(date) #######\n" >> ./logs/UnIMP_delta.txt
datasets=("chess" "shuttle" "power_consumption")
delta_values=(0.1 0.5 1 2 4 10)
for dataset in "${datasets[@]}"
do
    for delta in "${delta_values[@]}"
    do
        CUDA_VISIBLE_DEVICES=7 python3 -u main.py --epochs 1000 --header_type Linear --chunk_size 512 --chunk_batch 64 --device 0 --eval_epoch_gap 100 --mode finetune --data $dataset --load_model_name Linear_Epoch3999_allinone_v3 --delta $delta >> ./logs/UnIMP_delta.txt 2>&1
        echo "Finished processing dataset: $dataset, $delta" >> ./logs/UnIMP_delta.txt
    done
done &