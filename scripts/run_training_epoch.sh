echo "\n####### running time is $(date) #######\n" >> ./logs/UnIMP_training_epoch.txt
datasets=("parkinsons")
# missing_rates=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8)
model_names=("Linear_Epoch999_allinone_v3" "Linear_Epoch1999_allinone_v3" "Linear_Epoch2999_allinone_v3" "Linear_Epoch3999_allinone_v3")
for dataset in "${datasets[@]}"
do
    for model_name in "${model_names[@]}"
    do
        CUDA_VISIBLE_DEVICES=1 python3 -u main.py --epochs 1000 --header_type Linear --chunk_size 512 --chunk_batch 64 --device 0 --missing_ratio 0.2 --mode testing --missing_mechanism MCAR --data "$dataset" --load_model_name "$model_name" >> ./logs/UnIMP_training_epoch.txt 2>&1
        echo "Finished processing dataset: $dataset with missing rate: $missing_rate" >> ./logs/UnIMP_training_epoch.txt
    done
done &