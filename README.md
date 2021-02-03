## Face shadow creator and remover

Combination of two other repositories: https://github.com/zhhoper/DPR and 
https://github.com/kampta/face-seg and some OpenCV code.

Create face shadow adds a shadow on the face and slight shadow on the hair, but leaves the background untouched.
Sample images:

<p align="center">
  <img src="https://user-images.githubusercontent.com/25400249/106814189-5008f000-6672-11eb-9430-7e21af3987af.png" width="200"/>
  <img src="https://user-images.githubusercontent.com/25400249/106814074-28198c80-6672-11eb-8fdb-ceb969153643.png" width="200"/>
  <img src="https://user-images.githubusercontent.com/25400249/106814126-38ca0280-6672-11eb-86b2-fd82b46aa0d5.png" width="200"/>
</p>

Usage: 

```
python create_face_shadow.py --input_data_folder ./data --output_folder ./result
```

Shadow remover to come soon.

For light modification model credits to:
```
@InProceedings{DPR,
  title={Deep Single Portrait Image Relighting},
  author = {Hao Zhou and Sunil Hadap and Kalyan Sunkavalli and David W. Jacobs},
  booktitle={International Conference on Computer Vision (ICCV)},
  year={2019}
}
```
