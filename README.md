# Towards High-Fidelity Face Swapping: A Comprehensive Survey and New Benchmark

Project page: https://casia-nlprai.github.io/face-swapping-survey/

## Dataset & Weights

### Face Images & Videos

The face images (`faces/`) and raw videos (`videos/`) are available via Baidu Pan. To obtain the download password, you must:

1. Download and fill out the [**CASIA FaceSwapping Dataset License Agreement**](./CASIA%20FaceSwapping%20Dataset%20License%20Agreement.docx).
2. Send the **signed agreement** to  **cripac_3dface@cripac.ia.ac.cn**, with a copy to **qi.li@mais.ia.ac.cn**.
3. The password will be sent to you via email.

Baidu Pan: https://pan.baidu.com/s/1yaIUvP4gIjTspanO4QQqdw

```
CASIA_FaceSwapping/
├── faces/                    # Aligned face images ({video_id}_{frame}.png)
└── videos/                   # Original raw videos ({video_id}.mp4)
```

### Pretrained Models

Pretrained model weights and pre-computed identity features are available via Baidu Pan (password: `6vkk`).

Baidu Pan: https://pan.baidu.com/s/1fXdbuc_qfOLQjdjGwjSQLg

```
pretrained_models/
├── faces_landmarks/          # 5-point landmarks (x y per line, 5 lines per file)
├── CosFace_ACC99.28.pth
├── phase1_wpdc_vdc.pth.tar
├── hopenet_robust_alpha1.pkl
├── th_model_params.pth
├── pt_inception-2015-12-05-6726825d.pth
├── BFM_mSEmTFK68etc.chj
├── BFM_similarity_Lm3D_all.mat
└── features_dict.npy            # Pre-computed identity features for ID Retrieval
```

## Repository Structure

```
face-swapping-survey/
├── benchmark/                        # Protocol pair files and annotations
│   ├── image_pairs_normal.txt        # Protocol 1 image pairs (4,500)
│   ├── image_pairs_cross_ethnicity.txt  # Protocol 2 image pairs (1,200)
│   ├── image_pairs_cross_attribute.txt  # Protocol 3 image pairs (4,300)
│   ├── video_pairs_normal.txt        # Protocol 1 video pairs (30)
│   ├── video_pairs_cross_ethnicity.txt  # Protocol 2 video pairs (30)
│   └── video_pairs_cross_attribute.txt  # Protocol 3 video pairs (30)
├── eval_faceswap/                    # Evaluation code
│   ├── main.py                       # Entry point
│   ├── config.py                     # Path configuration
│   ├── eval_id_retrieval.py          # ID Retrieval
│   ├── eval_id_similarity.py         # ID Similarity
│   ├── eval_pose_err.py              # Pose Error (Hopenet)
│   ├── eval_exp_3ddfa.py             # Expression Error (3DDFA)
│   ├── eval_exp_facewarehouse.py     # Expression Error (FaceWarehouse)
│   ├── eval_FID.py                   # FID
│   ├── eval_SSIM.py                  # SSIM
│   ├── prepare_id_features.py        # Pre-compute identity features
│   ├── face_recognition/             # CosFace
│   ├── pose_estimation/              # Hopenet
│   ├── DDFA/                         # 3DDFA
│   ├── facewarehouse/                # FaceWarehouse + BFM
│   ├── pytorch_fid_new/              # FID module
│   └── id_prepare/                   # Identity features storage
├── index.html
└── assets/
```

## Evaluation Protocols

| Protocol | Ethnicity | Attributes | Image Pairs | Video Pairs | Focus |
|----------|-----------|------------|-------------|-------------|-------|
| Normal | Same | Normal | 4,500 | 30 | Baseline performance |
| Cross-ethnicity | Different | Normal | 1,200 | 30 | Demographic generalization |
| Cross-attribute | Same | Different | 4,300 | 30 | Robustness to pose/expression/lighting |

### Pair File Format

**Image pairs** (`benchmark/image_pairs_*.txt`): each line is `{source} {target}`, e.g. `0041-0_0.png 1268-0_0.png`. Pairs are bidirectional.

**Video pairs** (`benchmark/video_pairs_*.txt`): each line is `{source_video} {target_video}`, e.g. `1271-0 0113-1`. For video swapping, extract frame 0 from source as identity, frames 0-99 from target.

## Getting Started

### 1. Setup

```bash
git clone https://github.com/casia-nlprai/face-swapping-survey.git
cd face-swapping-survey

pip install torch torchvision opencv-python numpy scipy Pillow tqdm pytorch-msssim
```

Download data from Baidu Pan, then copy model weights and identity features:

```bash
DL=/path/to/CASIA_FaceSwapping

cp $DL/pretrained_models/CosFace_ACC99.28.pth        eval_faceswap/face_recognition/
cp $DL/pretrained_models/phase1_wpdc_vdc.pth.tar     eval_faceswap/DDFA/models/
cp $DL/pretrained_models/hopenet_robust_alpha1.pkl   eval_faceswap/pose_estimation/
cp $DL/pretrained_models/th_model_params.pth         eval_faceswap/facewarehouse/network/
cp $DL/pretrained_models/pt_inception-2015-12-05-6726825d.pth eval_faceswap/
cp $DL/pretrained_models/BFM_mSEmTFK68etc.chj        eval_faceswap/facewarehouse/BFM/
cp $DL/pretrained_models/BFM_similarity_Lm3D_all.mat eval_faceswap/facewarehouse/BFM/
cp $DL/pretrained_models/features_dict.npy            eval_faceswap/id_prepare/
```

Configure paths in `eval_faceswap/config.py`:

```python
root = '/path/to/swap_results/'              # Swapped images root (also where eval results are saved)
ori_data_root = '/path/to/CASIA_FaceSwapping/'  # Downloaded data (must contain faces/ and faces_landmarks/)
```

Or use environment variables: `EVAL_RESULT_ROOT`, `EVAL_ORI_DATA_ROOT`.

### 2. Run Face Swapping

Read the pair files in `benchmark/` (e.g. `image_pairs_normal.txt`). For each line, the left image is the **source** (identity), the right is the **target** (attributes). Run your face swapping method and save results as:

```
swap_results/{method_name}/{swap_type}/{source_image_name}-{target_image_name}.png
```

- `{swap_type}`: `all_simple` (Normal), `all_cross_ethnicity`, or `all_cross_attribute`
- Example: source `0041-0_0.png` + target `1268-0_0.png` → `0041-0_0-1268-0_0.png`
- The file name is the source image name and target image name joined by `-`. The evaluation code parses names by splitting at the boundary between `_digit` and `XXXX` (e.g., `0041-0_0` `-` `1268-0_0`).

### 3. Run Evaluation

```bash
cd eval_faceswap

# Evaluate one method on all protocols
python main.py --methods SimSwap --gpu 0

# Evaluate specific protocols (all_simple = Normal in paper)
python main.py --methods SimSwap --types all_simple all_cross_ethnicity --gpu 0
```

**Arguments:** `--methods` (required), `--types` (default: all three), `--gpu` (default: 0)

Results saved as `eval_results_{method}_{type}.txt`.

For video evaluation, we measure **Subject Consistency** and **Background Consistency** via [VBench](https://github.com/Vchitect/VBench):

```bash
pip install vbench
vbench evaluate --dimension subject_consistency --videos_path /path/to/swapped_videos/ --mode=custom_input
vbench evaluate --dimension background_consistency --videos_path /path/to/swapped_videos/ --mode=custom_input
```

Image-level metrics (ID Retrieval, Pose Error, etc.) can also be applied per-frame to videos.

## Metrics

| Metric | Description |
|--------|-------------|
| ID Retrieval ↑ | Top-1 identity retrieval accuracy (CosFace) |
| ID Similarity ↑ | Cosine similarity between source and swapped features |
| Pose Error ↓ | L2 distance of head pose angles (Hopenet) |
| Expression Error ↓ | L2 distance of expression parameters (3DDFA / FaceWarehouse) |
| FID ↓ | Frechet Inception Distance |
| SSIM ↑ | Structural Similarity |
| Subject Consistency ↑ | Temporal identity consistency ([VBench](https://github.com/Vchitect/VBench)) |
| Background Consistency ↑ | Temporal background consistency ([VBench](https://github.com/Vchitect/VBench)) |

## Citation

```bibtex
@article{li2026highfidelityfaceswapping,
  title   = {Towards High-Fidelity Face Swapping: A Comprehensive Survey and New Benchmark},
  author  = {Li, Qi and Wang, Weining and Du, Shuangjun and Peng, Bo and Dong, Jing and Wang, Kun and Sun, Zhenan and Yang, Ming-Hsuan},
  journal = {Pending},
  year    = {2026}
}
```
