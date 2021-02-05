# Face shadow creator and remover

## Shadow creator

Combination of two other repositories: https://github.com/zhhoper/DPR and 
https://github.com/kampta/face-seg and some OpenCV code.

Create face shadow adds a shadow on the face and slight shadow on the hair, but leaves the background untouched.
Sample images:

<p align="center">
  <img src="https://user-images.githubusercontent.com/25400249/106814189-5008f000-6672-11eb-9430-7e21af3987af.png" width="180"/>
  <img src="https://user-images.githubusercontent.com/25400249/106814074-28198c80-6672-11eb-8fdb-ceb969153643.png" width="180"/>
  <img src="https://user-images.githubusercontent.com/25400249/106814126-38ca0280-6672-11eb-86b2-fd82b46aa0d5.png" width="180"/>
  <img src="https://user-images.githubusercontent.com/25400249/107083182-8de25180-67f5-11eb-81de-2f68be1df6fa.png" width="180"/>
</p>

Usage: 

```
python create_face_shadow.py --input_data_folder ./data --output_folder ./result
```

## Shadow remover

<p align="center">
  <img src="https://user-images.githubusercontent.com/25400249/107082930-2b895100-67f5-11eb-9714-ec4a65aa0801.jpg" width="180"/>
  <img src="https://user-images.githubusercontent.com/25400249/107082913-262c0680-67f5-11eb-9788-157616721bb1.jpg" width="180"/>
  <img src="https://user-images.githubusercontent.com/25400249/107082933-2c21e780-67f5-11eb-83d4-ff56bfeb03a9.jpg" width="180"/>
  <img src="https://user-images.githubusercontent.com/25400249/107082916-26c49d00-67f5-11eb-8cbe-4c21ad42141a.jpg" width="180"/>
</p>


Usage: 

```
python deshadow.py --input_data_folder ./data/shadowed --output_folder ./result/deshadowed
```

For light modification model and code credits to:
```
@InProceedings{DPR,
  title={Deep Single Portrait Image Relighting},
  author = {Hao Zhou and Sunil Hadap and Kalyan Sunkavalli and David W. Jacobs},
  booktitle={International Conference on Computer Vision (ICCV)},
  year={2019}
}
```
For face segmentation model and code credits to: [Kamal Gupta](https://kampta.github.io/)