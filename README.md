# 1. Env Set Up

## 1.1. Install Isaac Sim 5.0.0 and verify

```bash
conda create -n eval_gr00t_16 python=3.11
conda activate eval_gr00t_16
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com
isaacsim
```

## 1.2. Install Isaac Lab 2.2.0 and verify

```bash
cd IsaacLab-2.2.0
./isaaclab.sh --install
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
```

## 1.3. Install GR00T n1.6 and verify

```bash
cd IsaacLabEvalTasks/source/isaaclab_eval_tasks
python -m pip install -U pip setuptools wheel ninja packaging
python -m pip install -e .
python -m pip uninstall -y flash-attn
python -m pip install --no-build-isolation --no-binary flash-attn "flash-attn==2.7.4.post1"
export PYTHONPATH=$PYTHONPATH:$INSTALL_DIR/IsaacLabEvalTasks/submodules/Isaac-GR00T-N1.6
python -c "import gr00t; print('gr00t imported successfully')"
python -c "import flash_attn; print('flash_attn import OK')"
```

## 1.4. Register Tasks

```bash
cd IsaacLabEvalTasks
python -m pip install -e source/isaaclab_eval_tasks
```

# 2. Deploy GR00T n1.6

Checkpoints can be downloaded from https://huggingface.co/YifWRobotics/Mar4-Gr00t-n1-6-Exhaust-Pipe-Sorting-task (for Exhaust-Pipe-Sorting-task) and https://huggingface.co/YifWRobotics/Mar4-Gr00t-n1-6-Nut-Pouring-task (for Nut-Pouring-task).

## 2.1. Eval Exhaust Pipe Sorting Task

```bash
export EVAL_RESULTS_FNAME="./eval_gr1_exhause_pipe_sorting.json"
python scripts/evaluate_gn1.py \
  --action_horizon 50 \
  --num_feedback_actions 16 \
  --num_envs 4 \
  --task_name pipesorting \
  --embodiment_tag new_embodiment \
  --eval_file_path $EVAL_RESULTS_FNAME \
  --checkpoint_name gr00t-n1-2b-tuned-pipesorting \
  --model_path PATH TO Exhaust-Pipe-Sorting-task/checkpoint-20000 \
  --rollout_length 30 \
  --seed 10 \
  --max_num_rollouts 100
```

## 2.2. Eval Nut Pouring Task

```bash
export EVAL_RESULTS_FNAME="./eval_gr1_nut_pouring.json"
python scripts/evaluate_gn1.py \
  --action_horizon 50 \
  --num_feedback_actions 16 \
  --num_envs 4 \
  --task_name nutpouring \
  --embodiment_tag new_embodiment \
  --eval_file_path $EVAL_RESULTS_FNAME \
  --checkpoint_name gr00t-n1-2b-tuned-nutpouring \
  --model_path PATH TO Nut-Pouring-task/checkpoint-20000 \
  --rollout_length 30 \
  --seed 10 \
  --max_num_rollouts 100
```
