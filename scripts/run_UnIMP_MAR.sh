echo "\n####### running time is $(date) #######\n" >> ./logs/UnIMP_MAR.txt
datasets=("parkinsons" "heart" "phishing" "bike" "chess" "shuttle" "power_consumption")
(
for dataset in "${datasets[@]}"
do
    CUDA_VISIBLE_DEVICES=0 python3 -u main.py --epochs 1000 --header_type Linear --chunk_size 512 --chunk_batch 64 --device 0 --eval_epoch_gap 100 --mode finetune --missing_mechanism MAR --data $dataset --load_model_name Linear_Epoch3999_allinone_v1 >> ./logs/UnIMP_MAR.txt 2>&1
    echo "Finished processing dataset: $dataset" >> ./logs/UnIMP_MAR.txt
done
) &
