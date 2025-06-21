# remove docker related
git rm envs/DockerfileOPPO
git rm scripts/installation/docker_env/build_oppo.sh
git rm scripts/installation/docker_env/run_oppo.sh

# remove data softlinks
git rm data/*
mkdir data
echo > data/__init__.py
git add data/__init__.py

# remove optional file
git rm scripts/evaluation/prepare_visualization_data.sh

# commit
git commit -m "code: clear files for release purpose"
