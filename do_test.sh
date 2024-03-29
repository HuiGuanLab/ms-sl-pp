collection=tvr
visual_feature=i3d_resnet
#collection=charades
#visual_feature=i3d_rgb_lgi
#collection=activitynet
#visual_feature=i3d
#collection=didemo
#visual_feature=rgb_flow
root_path=/home/cxk/pvr/VisualSearch
model_dir=tvr-pos_cat_32-2022_07_28_20_31_21

# training

python method/eval.py  --collection $collection --visual_feature $visual_feature \
                    --root_path $root_path  --dset_name $collection --model_dir $model_dir