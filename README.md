# Adaptive-Dual-domain-Learning-for-Underwater-Image-Enhancement

Welcome! This is the official implementation of the paper "[Adaptive Dual-domain Learning for Underwater Image Enhancement](https://ojs.aaai.org/index.php/AAAI/article/view/32692)".

[[📖 Paper](https://ojs.aaai.org/index.php/AAAI/article/view/32692)] [[Checkpoints](xxxxxxx)] [[LSUI Datasets](https://github.com/LintaoPeng/U-shape_Transformer_for_Underwater_Image_Enhancement)]

## 💥 News
- **[2025.04]** We release the training code and data for SS-UIE 🔧
- **[2024.12]** The conference paper is accepted by **AAAI 2025** 🎉
- **[2024.07]** The conference paper is submitted to AAAI 2025 🚀

## :art: Abstract

We propose a novel UIE method based on spatial-spectral dual-domain adaptive learning, termed SS-UIE. Specifically, we first introduce a spatial-wise Multi-scale Cycle Selective Scan (MCSS) module and a Spectral-Wise Self-Attention (SWSA) module, both with linear complexity, and combine them in parallel to form a basic Spatial-Spectral block (SS-block). Benefiting from the global receptive field of MCSS and SWSA, SS-block can effectively model the degradation levels of different spatial regions and spectral bands, thereby enabling degradation level-based dual-domain adaptive UIE. By stacking multiple SS-blocks, we build our SS-UIE network. Additionally, a Frequency-Wise Loss (FWL) is introduced to narrow the frequency-wise discrepancy and reinforce the model's attention on the regions with high-frequency details. Extensive experiments validate that the SS-UIE technique outperforms state-of-the-art UIE methods while requiring cheaper computational and memory costs.

<p align="center">
    <img src="figs/method.png" width="90%"> <br>
</p>

The main contributions of our paper are as follows:
1. Our proposed MCSS and SWSA module can obtain the spatial-wise and spectral-wise global receptive fields with linear complexity, respectively, thereby modeling the degradation levels in different spatial regions and spectral bands.
2. We combined MCSS and SWSA in parallel to form an SS-block, which can reinforce the network's attention to the spatial regions and spectral bands with serious attenuation, and achieve degradation level-based adaptive UIE.  
3. The proposed FWL function can narrow the frequency-wise discrepancy, and force the model to restore high-frequency details adaptively without additional memory and computational costs.

Our SS-UIE outperforms SOTA UIE methods in quantitative evaluation and visual comparison with cheaper computational and memory costs.

<p align="center">
    <img src="figs/fig2.jpg" width="100%"> <br>
</p>
  

## 💪 Get Started

### 1. Clone the repository:

   ```bash
   git clone https://github.com/LintaoPeng/SS-UIE.git
   cd SS-UIE
   ```

To set up the environment for this project, follow the steps below:

### 2. Create and Activate Conda Environment

```bash
conda create -n your_env_name python=3.10
conda activate your_env_name
```

### 3. Install PyTorch with CUDA Support

```bash
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 3. Install CUDA Compiler (nvcc)

```bash
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
```

### 4. Install Additional Dependencies

```bash
conda install packaging
pip install timm
pip install scikit-image
pip install opencv-python
pip install causal-conv1d==1.1.1
pip install mamba-ssm==1.1.1
```
 

### 🚀 Training 
#### Training ORM
To fine-tune the ORM model, run the following command:
```
bash scripts/orm_ft.sh
```
#### Training PARM
To train the PARM model, run the following command:
```
bash scripts/parm.sh
```
#### Training DPO
To train Show-o with DPO, run the following command:
```
bash scripts/dpo.sh 
```

### 📊 Evaluation              
#### 0. Baseline Model ([Show-o](https://github.com/showlab/Show-o)) 🎨
Run the following command to use the baseline model:
```
torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_port=12475 main.py \
--prompts_file geneval/prompts/generation_prompts.txt \
--metadata_file geneval/prompts/evaluation_metadata.jsonl \
--config config.yaml 
```
#### 1. Scaling Test-time Computation 📈

##### 1.1. Zero-shot ORM
Run the following command to use the zero-shot ORM:
```
torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_port=12475 main.py \
--prompts_file geneval/prompts/generation_prompts.txt \
--metadata_file geneval/prompts/evaluation_metadata.jsonl \
--config config.yaml \
--reward_model orm_zs 
```
##### 1.2. Fine-tuned ORM
Run the following command to use the fine-tuned ORM:
```
torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_port=12475 main.py \
--prompts_file geneval/prompts/generation_prompts.txt \
--metadata_file geneval/prompts/evaluation_metadata.jsonl \
--config config.yaml \
--reward_model orm_ft
```
##### 1.3. PARM
Run the following command to use PARM:
```
torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_port=12475 main.py \
--prompts_file geneval/prompts/generation_prompts.txt \
--metadata_file geneval/prompts/evaluation_metadata.jsonl \
--config config.yaml \
--reward_model parm 
```
#### 2. Preference Alignment with DPO 🔧

##### 2.1. Initial DPO
Run the following command to use intial DPO:
```
torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_port=12475 main.py \
--prompts_file geneval/prompts/generation_prompts.txt \
--metadata_file geneval/prompts/evaluation_metadata.jsonl \
--config config.yaml \
--dpo_model dpo
```
##### 2.2. Iterative DPO
Run the following command to use iterative DPO:
```
torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_port=12475 main.py \
--prompts_file geneval/prompts/generation_prompts.txt \
--metadata_file geneval/prompts/evaluation_metadata.jsonl \
--config config.yaml \
--dpo_model dpo_iter
```
##### 2.3. Iterative DPO with PARM Guidance
Run the following command to use iterative DPO with PARM guidance:
```
torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_port=12475 main.py \
--prompts_file geneval/prompts/generation_prompts.txt \
--metadata_file geneval/prompts/evaluation_metadata.jsonl \
--config config.yaml \
--dpo_model dpo_iter_parm_gudie
```
#### 3. Reasoning Strategy Integration 🧩

##### 3.1. Iterative DPO with PARM Guidance + PARM
Run the following command to combine iterative DPO with PARM guidance and PARM:
```
torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_port=12475 main.py \
--prompts_file geneval/prompts/generation_prompts.txt \
--metadata_file geneval/prompts/evaluation_metadata.jsonl \
--config config.yaml \
--reward_model parm \
--dpo_model dpo_iter_parm_gudie
```

## :white_check_mark: Citation

If you find this project useful for your research or applications, please kindly cite using this BibTeX:

```latex
@inproceedings{peng2025adaptive,
  title={Adaptive Dual-domain Learning for Underwater Image Enhancement},
  author={Peng, Lintao and Bian, Liheng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={6},
  pages={6461--6469},
  year={2025}
}
```


## 🧠 Related Work





















