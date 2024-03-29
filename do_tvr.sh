collection=tvr
visual_feature=i3d_resnet
map_size=32
q_feat_size=768
model_name=MS_SL_Net_pp
margin=0.1

root_path=/home/cxk/pvr/VisualSearch
device_ids=1
exp_id=$1
cluster_num=$2

# training
python method/train.py  --collection $collection --visual_feature $visual_feature \
                    --root_path $root_path  --dset_name $collection --exp_id $exp_id \
                    --map_size $map_size --q_feat_size $q_feat_size --model_name $model_name \
                    --margin $margin --device_ids $device_ids  \
                    --cluster --length_pos --cluster_num $cluster_num
