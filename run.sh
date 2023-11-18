CUDA_VISIBLE_DEVICE=1 python main.py 
# --synthetic \
# --train \
# --num 100 \
# --batch_size 32 \
# --degree 1 \
# --pi 10 \
# --lr 2e-4 --epoch 2000 \
# --condition non-linear --time_varying \
# --tol 0.05 \
# --max_d_L 4 --d_L 2 --d_X 8 \
# --sparsity_Bt 1e-4 --sparsity_Ct 0 --sparsity_Ct_1 1e-1 --DAG 1.0 \
# --graph_thres 0.2 \
# --seed 33 \
# --optimizer ADAM