# Jittor-Remdet
The jittor version of "RemDet: Rethinking Efficient Model Design for UAV Object Detection" [AAAI 2025] 
<details>
  <summary>
  <font size="+1">Abstract</font>
  </summary>
Object detection in Unmanned Aerial Vehicle (UAV) images has emerged as a focal area of research, which presents two significant challenges: i) objects are typically small and dense within vast images; ii) computational resource constraints render most models unsuitable for real-time deployment. Current real-time object detectors are not optimized for UAV images, and complex methods designed for small object detection often lack real-time capabilities. To address these challenges, we propose a novel detector, RemDet (Reparameter efficient multiplication Detector). Our contributions are as follows: 1) Rethinking the challenges of existing detectors for small and dense UAV images, and proposing information loss as a design guideline for efficient models. 2) We introduce the ChannelC2f module to enhance small object detection performance, demonstrating that high-dimensional representations can effectively mitigate information loss. 3) We design the GatedFFN module to provide not only strong performance but also low latency, effectively addressing the challenges of real-time detection. Our research reveals that GatedFFN, through the use of multiplication, is more cost-effective than feed-forward networks for high-dimensional representation. 4) We propose the CED module, which combines the advantages of ViT and CNN downsampling to effectively reduce information loss. It specifically enhances context information for small and dense objects. Extensive experiments on large UAV datasets, Visdrone and UAVDT, validate the real-time efficiency and superior performance of our methods. On the challenging UAV dataset VisDrone, our methods not only provided state-of-the-art results, improving detection by more than 3.4, but also achieve 110 FPS on a single 4090.
</details>


## Main results


#### notice:
Due to some exsisting differeces between these two frames, some codes are fixed, which may cause performences decay.Due to the reason flowing, use cpu to train model.
#### reason:
When trying using gpu an error come from jittor occur:
 **undefined symbol: _ZN6jittor11getDataTypeIdEE15cudnnDataType_tv**,but when using cpu there is no such error, and have known this happen in 
```python
class GradScaler:
  def step(...):
    ...
    optimizer.pre_step(loss)
```
when trying to get data to cpu ,the error occuered, and up to now a solution is still no found.
## Object Detection
### Environments
```shell
pip install jittor
pip install opencv-python
pip insatll pycocotools
python -m jittor_utils.install_cuda
```

### Dataset
The model trained using [VisDrone2019](https://github.com/VisDrone/VisDrone-Dataset) dataset
## Train
Train with only one GPUs:
```shell
python remdet.py
```
## Acknowledgements
To complete this project,I reference codes from:
- [Albumentations](https://github.com/albumentations-team/albumentations)
- [MMengine](https://github.com/open-mmlab/mmengine)
- [Pytorch](https://github.com/pytorch/pytorch)
- [RemDet](https://github.com/HZAI-ZJNU/RemDet)
- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [MMYOLO](https://github.com/open-mmlab/mmyolo)
### Framework
This project is completed under [jittor](https://github.com/Jittor/jittor) framework 
