## Train
Run run.ipynb with 32768 simStep each scenarios

## Test

```bash
cd app
```

### Build image

```bash
docker build -t energy-simulation .
```

### Run container

```bash
docker-compose run energy-simulation bash
```

### Set training mode to False in config.yaml to test 

```bash
./run_main_run_scenarios.sh /opt/mcr/R2025a | tee logs/output.log
```