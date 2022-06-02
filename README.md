
### Dataset

Our code is compatible with the dataset setup introduced by [Martinez et al.](https://github.com/una-dinosauria/3d-pose-baseline) and [Pavllo et al.](https://github.com/facebookresearch/VideoPose3D). Please refer to [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) to set up the Human3.6M dataset  (./data directory). 



### Train
```
. activate Pose3D
cd ../Pose3D
python run.py -k cpn_ft_h36m_dbb -f 27 -lr 0.0001 -lrd 0.99
python run.py -k cpn_ft_h36m_dbb -f 27 -lr 0.00004 -lrd 0.99
```

### Test
```
python run.py -k cpn_ft_h36m_dbb -c checkpoint --evaluate best_epoch.bin --render --viz-subject S11 --viz-action Walking --viz-camera 0 --viz-output output.gif --viz-size 3 --viz-downsample 2 --viz-limit 60 -f 27
```