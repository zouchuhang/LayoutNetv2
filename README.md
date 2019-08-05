# LayoutNetv2
PyTorch implementation for LayoutNetv2

Original Torch implementation of LayoutNet [here](https://github.com/zouchuhang/LayoutNet).

## Improvements upon LayoutNet
- Extend to general Manhattan layout (on our newly labeled [Matterport3D dataset]())
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

## Preporcess

## Training
- On PanoContext (note that we use Stanford 2D-3D as additional data in this script):
    ```
    python train_PC.py
    ```
- On Stanford 2D-3D (note that we use PanoContext as additional data in this script):
    ```
    python train_stanford.py
    ```
- On Matterport3D
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
