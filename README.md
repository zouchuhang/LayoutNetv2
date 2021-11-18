# LayoutNet v2
PyTorch implementation of our IJCV paper: 

### **Manhattan Room Layout Reconstruction from a Single 360 image: A Comparative Study of State-of-the-art Methods**

https://arxiv.org/pdf/1910.04099.pdf

- **New:** Our [MatterportLayout](https://github.com/ericsujw/Matterport3DLayoutAnnotation) Annotation dataset is released!

<img src='figs/teasor.png' width=700>

Original Torch implementation for LayoutNet is [here](https://github.com/zouchuhang/LayoutNet).

## Improvements upon LayoutNet
- Extend to general Manhattan layout (on our newly labeled MatterportLayout dataset)
- Use ResNet encoder instead of SegNet encoder
- Training details and implementation details
- Gradient ascent based post optimization, revised from sunset1995's PyTorch [implementation](https://github.com/sunset1995/pytorch-layoutnet)
- Add [random stretching](https://github.com/sunset1995/HorizonNet) data augmentation

## Requirements
- Python 3
- PyTorch >= 0.4.0
- numpy, scipy, pickle, skimage, sklearn, random, cv2, shapely
- torchvision
- Matlab (for depth rendering)

## Download Data and Pre-trained Model
- Download [pre-trained models](https://drive.google.com/file/d/1gFQq83gL4crNRKPq_Qjx-YDYup5Wf_LE/view?usp=sharing) and put them under the ./model/ folder.
- Download [pre-processed PanoContext and Stanford 2D-3D dataset in .pkl form](https://drive.google.com/drive/folders/1s_uPGrTW_9It-grehVFi1j5DkqZag2Cv?usp=sharing) and put them under the ./data/ folder.
- Download gt from original [LayoutNet](https://github.com/zouchuhang/LayoutNet/tree/master/data) and the processed one by [sunset1995](https://drive.google.com/file/d/1e-MuWRx3T4LJ8Bu4Dc0tKcSHF9Lk_66C/view) and put them under the ./data/ folder
- (Optional) Download original LayoutNet's .t7 [file](https://drive.google.com/file/d/1400fSLme70jnTnsmkPk4YLDAtQA0mgTP/view) and put them under the ./data/ folder
- Download our newly labeled [MatterportLayout dataset](https://github.com/ericsujw/Matterport3DLayoutAnnotation) and put them under the ./data/ folder.
- Download [pre-computed depth maps](https://drive.google.com/file/d/1V85M_uQF9oULas_UU7AHXyJnJrjcXzEb/view?usp=sharing) from our models trained on MatterportLayout dataset and put them under the current folder. 

## Preporcess
- We've provided sample code to transform original LayoutNet's .t7 file to .pkl file for PyTorch
    ```
    python t72pkl.py
    ``` 

## Training
- On PanoContext (note that we use Stanford 2D-3D as additional data in this script):
    ```
    python train_PC.py
    ```
- On Stanford 2D-3D (note that we use PanoContext as additional data in this script):
    ```
    python train_stanford.py
    ```
- On MatterportLayout
    ```
    python train_matterport.py
    ```

## Evaluation
- On PanoContext (Corner error, pixel error and 3D IoU)
    ```
    python test_PC.py
    ```
- On Stanford 2D-3D (Corner error, pixel error and 3D IoU)
    ```
    python test_stanford.py
    ```
- On Matterport3D (3D IoU, 2D IoU on under top-down view, RMSE for depth and delta\_1 for depth)
    ```
    python test_matterport.py
    ```
  For depth related evaluation, we need to render depth map from predicted corner position on equirectangualr view (you can skip this step as we've provided pre-computed depth maps from our approach)
  
  First, uncomment L313-L314 in test\_matterport.py, and comment out lines related to evaluation for depth. Run test\_matterport.py and save intermediate corner predictions to folder ./result\_gen. Then open matlab:
    ```
    cd matlab
    cor2depth
    cd ..
    ```
    Rendered depth maps will be saved to folder ./result\_gen\_depth/.
    Then comment out L313-L314 in test\_matterport.py, uncomment lines related to evaluation for depth, and run test\_matterport.py again
    
## Citation
Please cite our paper for any purpose of usage.
```
@article{zou2021manhattan,
  title={Manhattan Room Layout Reconstruction from a Single $ $360\^{}$\{$$\backslash$circ$\}$ $$360∘ Image: A Comparative Study of State-of-the-Art Methods},
  author={Zou, Chuhang and Su, Jheng-Wei and Peng, Chi-Han and Colburn, Alex and Shan, Qi and Wonka, Peter and Chu, Hung-Kuo and Hoiem, Derek},
  journal={International Journal of Computer Vision},
  volume={129},
  number={5},
  pages={1410--1431},
  year={2021},
  publisher={Springer}
}
```
