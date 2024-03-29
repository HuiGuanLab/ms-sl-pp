collection=charades
visual_feature=i3d_rgb_lgi
map_size=32
model_name=MS_SL_Net_pp
exp_id=$1
cluster_num=$2
root_path=/home/cxk/workplace/VisualSearch
device_ids=2
# training
python method/train.py  --collection $collection --visual_feature $visual_feature \
                      --root_path $root_path  --dset_name $collection --exp_id $exp_id \
                        --map_size $map_size --model_name $model_name --device_ids $device_ids\
                        --cluster --length_pos --cluster_num $cluster_num
