cd /jizhi/jizhi2/worker/trainer/youngeegu/projects/H2O/code
CUDA_VISIBLE_DEVICES=0,1,2,3  nohup   python -m torch.distributed.run  --nproc_per_node=4   \
    run.py \
    --mode test \
    --debug False \
    --test_dir  ./example/TENX92.h5 \
    --model_path ./example/best_epoch.pth \
    --batch_size 80 \
    --save_dir ./results/STain_hest1k/ \
    --CLIP True \
    --nbrs True \
    --FiLM True \
    > inference.log  2>&1 & 
