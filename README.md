```bash
cd app
docker build -t energy-simulation .
docker-compose run energy-simulation bash
./run_main_run_scenarios.sh /opt/mcr/R2025a
```