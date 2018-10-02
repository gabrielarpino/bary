export AWS_PROFILE='GPForecasting@Invenia'
docker build -t gabrieldamec:latest .
$(aws ecr get-login --no-include-email --registry-ids 052722095006 --region us-east-1)
REPO=$(grep ecr_uri resources.yml | cut -d' ' -f2)
docker tag gabrieldamec:latest ${REPO}:latest
docker push ${REPO}:latest
job_definition=$(grep job_definition resources.yml | cut -d' ' -f2)
manager_queue=$(grep manager_queue_arn resources.yml | cut -d' ' -f2)

aws batch submit-job --job-definition $job_definition --job-queue $manager_queue --job-name DAMEC_Experiment_2_g0_linex_d2 --container-overrides command="["julia", "--compilecache=no", "scripts/experiment.jl", "DAMEC_Experiment_2_g0_linex_d2", "-c", "batch", "-n", "2", "-a", "s3", "-s", "DAMEC_experiment_2_g0.jl"]"
aws batch submit-job --job-definition $job_definition --job-queue $manager_queue --job-name DAMEC_Experiment_2_g1_linex_d2 --container-overrides command="["julia", "--compilecache=no", "scripts/experiment.jl", "DAMEC_Experiment_2_g1_linex_d2", "-c", "batch", "-n", "2", "-a", "s3", "-s", "DAMEC_experiment_2_g1.jl"]"
aws batch submit-job --job-definition $job_definition --job-queue $manager_queue --job-name DAMEC_Experiment_2_g2_linex_d2 --container-overrides command="["julia", "--compilecache=no", "scripts/experiment.jl", "DAMEC_Experiment_2_g2_linex_d2", "-c", "batch", "-n", "2", "-a", "s3", "-s", "DAMEC_experiment_2_g2.jl"]"
aws batch submit-job --job-definition $job_definition --job-queue $manager_queue --job-name DAMEC_Experiment_2_g3_linex_d2 --container-overrides command="["julia", "--compilecache=no", "scripts/experiment.jl", "DAMEC_Experiment_2_g3_linex_d2", "-c", "batch", "-n", "2", "-a", "s3", "-s", "DAMEC_experiment_2_g3.jl"]"
aws batch submit-job --job-definition $job_definition --job-queue $manager_queue --job-name DAMEC_Experiment_2_g4_linex_d2 --container-overrides command="["julia", "--compilecache=no", "scripts/experiment.jl", "DAMEC_Experiment_2_g4_linex_d2", "-c", "batch", "-n", "2", "-a", "s3", "-s", "DAMEC_experiment_2_g4.jl"]"
aws batch submit-job --job-definition $job_definition --job-queue $manager_queue --job-name DAMEC_Experiment_2_g5_linex_d2 --container-overrides command="["julia", "--compilecache=no", "scripts/experiment.jl", "DAMEC_Experiment_2_g5_linex_d2", "-c", "batch", "-n", "2", "-a", "s3", "-s", "DAMEC_experiment_2_g5.jl"]"
