echo "\n####### running time is $(date) #######\n" >> ./logs/UnIMP_Missing_Rate.txt
datasets=("power_consumption")
missing_rates=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8)
for dataset in "${datasets[@]}"
do
    for missing_rate in "${missing_rates[@]}"
    do
        CUDA_VISIBLE_DEVICES=1 python3 -u main.py --epochs 100 --header_type Linear --chunk_size 512 --chunk_batch 64 --device 0 --eval_epoch_gap 100 --missing_ratio $missing_rate --mode finetune --missing_mechanism MCAR --data "$dataset" --load_model_name Linear_Epoch3999_allinone_v3 >> ./logs/UnIMP_Missing_Rate.txt 2>&1
        echo "Finished processing dataset: $dataset with missing rate: $missing_rate" >> ./logs/UnIMP_Missing_Rate.txt
    done
done &