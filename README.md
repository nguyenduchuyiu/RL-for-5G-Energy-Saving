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