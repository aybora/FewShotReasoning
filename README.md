# Few-Shot Vision-Language Reasoning for Satellite Imagery via Verifiable Rewards

## ICCV 2025 | Curated Data for Efficient Learning Workshop

We present the first few-shot reinforcement learning with verifiable reward (RLVR) framework for satellite imagery that eliminates the need for caption supervision--relying solely on lightweight, rule-based binary or IoU-based rewards. Adapting the "1-shot RLVR" paradigm from language models to vision-language models, we employ policy-gradient optimization with as few as one curated example to align model outputs for satellite reasoning tasks. 

### [Paper (arXiv)](https://arxiv.org/abs/2507.21745)

### Installation

```shell
git clone https://github.com/aybora/FewShotReasoning
conda create -n train python=3.10 -y
conda activate train
cd ~/FewShotReasoning/train
pip3 install -e ".[dev]"
pip3 install wandb==0.18.3
pip3 install flash-attn==2.7.3 --no-build-isolation
```
### Training

Below script works on at least one node with 4 x H100s or A100s (65-80 GB).

```shell
export WANDB_RUN_NAME=Qwen-VL-2B-GRPO-$(date +%Y-%m-%d-%H-%M-%S)
torchrun \
    --nproc_per_node="$GPUS_PER_NODE" \
    --nnodes="$SLURM_NNODES" \
    --node_rank="$SLURM_NODEID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
    --rdzv_id $SLURM_JOB_ID \
    src/open_r1/grpo.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir checkpoints/${WANDB_RUN_NAME} \
    --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
    --dataset_name aybora/VHM_dataset_grpo_cls_vqa_vg_2k_example \
    --max_prompt_length 8192 \
    --max_completion_length 8192 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 true \
    --beta 0.001 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 2359296 \
    --save_total_limit 6 \
    --num_train_epochs 64 \
    --num_generations 4 \
    --save_steps 100 \
    --run_name $WANDB_RUN_NAME
```
You may need to adjust some of the parameters (MASTER_ADDR, GPUS_PER_NODE etc.) depending on your multi-gpu, multi-node setting. 

Below, you may select the dataset to train, or directly use the models. 

<table>
  <thead>
    <tr style="text-align: right;">
      <th>name</th>
      <th>model</th>
      <th>dataset</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>$\pi_{1V}$</td>
      <td><a href="https://huggingface.co/aybora/Qwen2-VL-2B-VQA-1-EX">aybora/Qwen2-VL-2B-VQA-1-EX</a></td>
      <td><a href="https://huggingface.co/datasets/aybora/VHM_dataset_grpo_vqa_1_example">aybora/VHM_dataset_grpo_vqa_1_example</a></td>
    </tr>
    <tr>
      <td>$\pi_{1C}$</td>
      <td><a href="https://huggingface.co/aybora/Qwen2-VL-2B-CLS-1-EX">aybora/Qwen2-VL-2B-CLS-1-EX</a></td>
      <td><a href="https://huggingface.co/datasets/aybora/VHM_dataset_grpo_cls_1_example">aybora/VHM_dataset_grpo_cls_1_example</a></td>
    </tr>
    <tr>
      <td>$\pi_{1G}$</td>
      <td><a href="https://huggingface.co/aybora/Qwen2-VL-2B-VG-1-EX">aybora/Qwen2-VL-2B-VG-1-EX</a></td>
      <td><a href="https://huggingface.co/datasets/aybora/VHM_dataset_grpo_vg_1_example">aybora/VHM_dataset_grpo_vg_1_example</a></td>
    </tr>
    <tr>
      <td>$\pi_{2VC}$</td>
      <td><a href="https://huggingface.co/aybora/Qwen2-VL-2B-CLS-VQA-2-EX">aybora/Qwen2-VL-2B-CLS-VQA-2-EX</a></td>
      <td><a href="https://huggingface.co/datasets/aybora/VHM_dataset_grpo_cls_vqa_2_example">aybora/VHM_dataset_grpo_cls_vqa_2_example</a></td>
    </tr>
    <tr>
      <td>$\pi_{2G}$</td>
      <td><a href="https://huggingface.co/aybora/Qwen2-VL-2B-VG-2-EX">aybora/Qwen2-VL-2B-VG-2-EX</a></td>
      <td><a href="https://huggingface.co/datasets/aybora/VHM_dataset_grpo_vg_2_example">aybora/VHM_dataset_grpo_vg_2_example</a></td>
    </tr>
    <tr>
      <td>$\pi_{4VC}$</td>
      <td><a href="https://huggingface.co/aybora/Qwen2-VL-2B-CLS-VQA-4-EX">aybora/Qwen2-VL-2B-CLS-VQA-4-EX</a></td>
      <td><a href="https://huggingface.co/datasets/aybora/VHM_dataset_grpo_cls_vqa_4_example">aybora/VHM_dataset_grpo_cls_vqa_4_example</a></td>
    </tr>
    <tr>
      <td>$\pi_{4VCG}$</td>
      <td><a href="https://huggingface.co/aybora/Qwen2-VL-2B-CLS-VQA-VG-4-EX">aybora/Qwen2-VL-2B-CLS-VQA-VG-4-EX</a></td>
      <td><a href="https://huggingface.co/datasets/aybora/VHM_dataset_grpo_cls_vqa_vg_4_example">aybora/VHM_dataset_grpo_cls_vqa_vg_4_example</a></td>
    </tr>
    <tr>
      <td>$\pi_{8VC}$</td>
      <td><a href="https://huggingface.co/aybora/Qwen2-VL-2B-CLS-VQA-8-EX">aybora/Qwen2-VL-2B-CLS-VQA-8-EX</a></td>
      <td><a href="https://huggingface.co/datasets/aybora/VHM_dataset_grpo_cls_vqa_8_example">aybora/VHM_dataset_grpo_cls_vqa_8_example</a></td>
    </tr>
    <tr>
      <td>$\pi_{8VCG}$</td>
      <td><a href="https://huggingface.co/aybora/Qwen2-VL-2B-CLS-VQA-VG-8-EX">aybora/Qwen2-VL-2B-CLS-VQA-VG-8-EX</a></td>
      <td><a href="https://huggingface.co/datasets/aybora/VHM_dataset_grpo_cls_vqa_vg_8_example">aybora/VHM_dataset_grpo_cls_vqa_vg_8_example</a></td>
    </tr>
    <tr>
      <td>$\pi_{16VC}$</td>
      <td><a href="https://huggingface.co/aybora/Qwen2-VL-2B-CLS-VQA-16-EX">aybora/Qwen2-VL-2B-CLS-VQA-16-EX</a></td>
      <td><a href="https://huggingface.co/datasets/aybora/VHM_dataset_grpo_cls_vqa_16_example">aybora/VHM_dataset_grpo_cls_vqa_16_example</a></td>
    </tr>
    <tr>
      <td>$\pi_{16VC}$ - 7B</td>
      <td><a href="https://huggingface.co/aybora/Qwen2-VL-7B-CLS-VQA-16-EX">aybora/Qwen2-VL-7B-CLS-VQA-16-EX</a></td>
      <td><a href="https://huggingface.co/datasets/aybora/VHM_dataset_grpo_cls_vqa_16_example">aybora/VHM_dataset_grpo_cls_vqa_16_example</a></td>
    </tr>
    <tr>
      <td>$\pi_{32VCG}$</td>
      <td><a href="https://huggingface.co/aybora/Qwen2-VL-2B-CLS-VQA-VG-32-EX">aybora/Qwen2-VL-2B-CLS-VQA-VG-32-EX</a></td>
      <td><a href="https://huggingface.co/datasets/aybora/VHM_dataset_grpo_cls_vqa_vg_32_example">aybora/VHM_dataset_grpo_cls_vqa_vg_32_example</a></td>
    </tr>
    <tr>
      <td>$\pi_{64VCG}$</td>
      <td><a href="https://huggingface.co/aybora/Qwen2-VL-2B-CLS-VQA-VG-64-EX">aybora/Qwen2-VL-2B-CLS-VQA-VG-64-EX</a></td>
      <td><a href="https://huggingface.co/datasets/aybora/VHM_dataset_grpo_cls_vqa_vg_64_example">aybora/VHM_dataset_grpo_cls_vqa_vg_64_example</a></td>
    </tr>
    <tr>
      <td>$\pi_{128VCG}$</td>
      <td><a href="https://huggingface.co/aybora/Qwen2-VL-2B-CLS-VQA-VG-128-EX">aybora/Qwen2-VL-2B-CLS-VQA-VG-128-EX</a></td>
      <td><a href="https://huggingface.co/datasets/aybora/VHM_dataset_grpo_cls_vqa_vg_128_example">aybora/VHM_dataset_grpo_cls_vqa_vg_128_example</a></td>
    </tr>
    <tr>
      <td>VHM-RL</td>
      <td><a href="https://huggingface.co/aybora/Qwen2-VL-2B-CLS-VQA-VG-2000-EX">aybora/Qwen2-VL-2B-CLS-VQA-VG-2000-EX</a></td>
      <td><a href="https://huggingface.co/datasets/aybora/VHM_dataset_grpo_cls_vqa_vg_2k_example">aybora/VHM_dataset_grpo_cls_vqa_vg_2k_example</a></td>
    </tr>
  </tbody>
</table>

### Evaluation

For evaluation, you need to install a couple more packages. To have a better reproducibility, it is suggested to create a new environment for evaluation.

```shell
  conda create -n eval python==3.10 -y
  conda activate eval
  cd eval 
  bash basic_env_setup.sh
```

First download datasets eval folder from our [Huggingface Repo](https://huggingface.co/datasets/aybora/scorers_datasets_eval), which is forked from [ScoreRS HF Repo](https://huggingface.co/datasets/LHRS/RSRM/tree/main).

To evaluate our, or your reproduced model, you may use the script below:

```shell

  SCRIPT_PATH=./FewShotReasoning/eval/python_script/evaluation/rs_evaluation.py
  DATA_ROOT="Your path to datasets eval folder"
  OUTPUT_DIR="Your path to eval log file"
  model_type=lmdeploy
  MODEL_PATH=aybora/Qwen2-VL-2B-CLS-VQA-16-EX
  REASONING_CONFIG=./FewShotReasoning/eval/config/qwen2_thinking_template.json

  PYTHONPATH=$(pwd) CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --mixed_precision bf16 $SCRIPT_PATH \
      --data_root $DATA_ROOT \
      --output_dir $OUTPUT_DIR \
      --model_type $model_type \
      --model_path $MODEL_PATH \
      --force_inference true \
      --task all \
      --reasoning_config $REASONING_CONFIG

```

### Acknowledgements

Our work is derived from [Qwen2-VL](https://github.com/QwenLM/Qwen2.5-VL) for the base model, [open-r1-multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal) for the forked main train code, [ScoreRS](https://github.com/NJU-LHRS/ScoreRS) for the forked main evaluation code and [VHM](https://github.com/opendatalab/VHM) for the dataset and the base captions. We appreciate all of these great works.

### Citation

If you find this code useful for your research, consider citing our work:

```bibtex
@misc{koksal2025fewshotvisionlanguagereasoningsatellite,
      title={Few-Shot Vision-Language Reasoning for Satellite Imagery via Verifiable Rewards}, 
      author={Aybora Koksal and A. Aydin Alatan},
      year={2025},
      eprint={2507.21745},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.21745}, 
}
```

