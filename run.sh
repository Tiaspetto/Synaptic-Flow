declare -a PrunerArray=("synflow" "snip" "grasp" "mag" "rand")
for seed in 0 1 2; do
	for val in "${PrunerArray[@]}"; do
		for d in 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 2.25 2.50 2.75 3.00; do
			if [ "$val" = "synflow" ]
			then
				CUDA_VISIBLE_DEVICES=0 python3 main.py --experiment=singleshot  --lr=0.0001 --lr-drops 60 120 --lr-drop-rate=0.1 --weight-decay=1e-4 --pre-epochs=0 --post-epochs=160 --gpu=0 --expid=sr_model_rcan_102032${val}comp_ratio${d}run${seed} --seed=$seed --dataset=div2k --model-class=srmodel --model=rcan --pruner=$val --prune-epochs=100 --compression=$d
			else
				CUDA_VISIBLE_DEVICES=0 python3 main.py --experiment=singleshot  --lr=0.0001 --lr-drops 60 120 --lr-drop-rate=0.1 --weight-decay=1e-4 --pre-epochs=0 --post-epochs=160 --gpu=0 --expid=sr_model_rcan_102032${val}comp_ratio${d}_run${seed} --seed=$seed --dataset=div2k --model-class=srmodel --model=rcan --prune-train-mode=True --pruner=$val --prune-epochs=1 --compression=$d
			fi
		done
	done
done
