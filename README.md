```bash
cd app
docker build -t energy-simulation .
docker-compose run energy-simulation bash
```

```bash
export ES_AUTO_CHECKPOINT=1
export ES_CHECKPOINT_PATH="/home/huy/Project/public_1/5GEnergySaving-Round1-public/logs/ppo_fast_model.pth"
export ES_TRAINING_MODE=0 # 0 if eval
./run_main_run_scenarios.sh /opt/mcr/R2025a | tee logs/output.log
```