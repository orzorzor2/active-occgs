# Active Language Reconstruction


## Docker usage
```
# detach docker container without losing the process (extra installed packages)
ctrl+p ctrl+q

# enter detached docker container
docker exec -it {DOCKER_CONTAINER_TAG} bash
```

## SplaTAM

```

pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

pip install charset_normalizer==2.0.4

```


## Generate ReplicaSLAM-Habiatat data for evaluation


```
# Full script
bash scripts/data/generate_replica_habitat.sh all

# individual
python src/data/update_ReplicaSLAM_data.py \
--cfg configs/Replica/office0/update_SLAM_data.py \
--seed 0 \
--result_dir results/tmp/generated_data \
--enable_vis 1
```


## Generate ReplicaSLAM-Habiatat NVS data for evaluation


```
# Full script
bash scripts/data/generate_replica_nvs.sh all

# individual
python src/data/generate_nvs_data.py \
--cfg configs/Replica/office0/generate_nvs_data.py \
--seed 0 \
--result_dir results/tmp/generated_data \
--enable_vis 1
```

## Run ActiveLang

```
# Replica
bash scripts/activelang/run_replica.sh {SCENE} {NUM_RUN} {EXP} {ENABLE_VIS} {GPU_ID}

实例指令:  bash scripts/activelang/run_replica.sh office0 1 ActiveGS 0 2

# Run Passive Splaatam
bash scripts/activelang/run_replica.sh office0 1 ActiveLang 0 1

# evaluation only
# bash scripts/evaluation/eval_splatam_recon.sh ${scene} ${i} ${EXP}
bash scripts/evaluation/eval_splatam_recon.sh office0 0 ActiveLang


```


