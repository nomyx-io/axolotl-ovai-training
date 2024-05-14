wandb_project=OVAI
model_base=llama3

run_id=$1
config_file=$2
output_dir="outputs/${model_base}"

accelerate launch -m axolotl.cli.train ${config_file} \
                                       --wandb_project ${wandb_project} \
                                       --wandb_run_id ${run_id} \
                                       --output_dir "${output_dir}/${run_id}" \
                                       --num_epochs 6