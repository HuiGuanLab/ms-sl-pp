# coding=utf-8
"""
An implementation of fast k-means, acceleration comes from batch operations

Reference:
	[1] https://github.com/DeMoriarty/fast_pytorch_kmeans/blob/master/fast_pytorch_kmeans/kmeans.py
"""
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def pairwise_distance(data1, data2, metric='euclidean',
						self_nearest=True, all_negative=False, p=2.0):
	"""
	pairwise distance
	Args:
		data1: 	torch.Tensor, [N1, L] or [B, N1, L]
		data2: 	torch.Tensor, [N2, L] or [B, N2, L]
		metric:	(str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
		self_nearest: If ture, make sure the closest point of each point is itself
		all_negative: If True, make sure the distance values are negative
	Return:
		a distance matrix [N1, N2] or [B, N1, N2]
	"""
	if metric == 'euclidean':
		dis = torch.cdist(data1, data2, p=p)


	elif metric == 'cosine':
		A_norm = data1 / (data1.norm(dim=-1, keepdim=True) + 1e-6)
		B_norm = data2 / (data2.norm(dim=-1, keepdim=True) + 1e-6)
		# A_norm = F.normalize(data1, dim=-1)
		# B_norm = F.normalize(data2, dim=-1)
		if data1.ndim == 3:
			dis = 1.0 - torch.bmm(A_norm, B_norm.transpose(-2, -1))
		else:
			dis = 1.0 - torch.matmul(A_norm, B_norm.transpose(-2, -1))
		# if data1.ndim == 3:
		# 	dis = torch.bmm(A_norm, B_norm.transpose(-2, -1))
		# else:
		# 	dis = torch.matmul(A_norm, B_norm.transpose(-2, -1))
		# for i in [9, 99, 299, 499]:
		# 	simi = dis[0,i,:].squeeze().cpu().numpy()
		# 	fig, ax = plt.subplots()
		# 	ax.plot([i for i in range(528)], simi)
		# 	plt.draw()
		# 	plt.savefig("/home/cxk/pvr_extend/ms-sl_cluster/pos_feat_cosine_simi_%s.jpg" % str(i))
		# 	plt.cla()
		# 	print(simi)
		# exit()

	else:
		raise NotImplementedError("{} metric is not implemented".format(metric))

	if all_negative:
		dis = dis - torch.max(dis) - 1.0

	if self_nearest:
		# avoid two same points
		diag = torch.arange(dis.shape[-1], device=dis.device, dtype=torch.long)
		dis[..., diag, diag] -= 1.0

	return dis


@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def KKZ_init(X, distance_matrix, K, batch=False):
	"""
	KKZ initilization for kmeans
	1. Choose the point with the maximum L2-norm as the first centroid.
	2. For j = 2, . . . ,K, each centroid μj is set in the following way: For
	any remaining data xi, we compute its distance di to the existing cen-
	troids. di is calculated as the distance between xi to its closest existing
	centroid. Then, the point with the largest di is selected as μj .

	Reference:
		I. Katsavounidis, C.-C. J. Kuo, and Z. Zhang. A new initialization tech-
		nique for generalized Lloyd iteration. IEEE Signal Processing Letters,
		1(10):144–146, 1994.

	"""
	l2_norm = torch.norm(X, dim=-1)
	if not batch:
		medoids = torch.arange(K, device=distance_matrix.device, dtype=torch.long)
		_, medoids[0] = torch.max(l2_norm, dim=0)
		for i in range(1, K):
			sub_dis_matrix = distance_matrix[:, medoids[:i]]
			# print(sub_dis_matrix.shape)
			values, indices = torch.min(sub_dis_matrix, dim=1)
			medoids[i] = torch.argmax(values, dim=0)

		# import pdb; pdb.set_trace()
		return medoids

	else:
		# batch version
		batch_i = torch.arange(X.shape[0], dtype=torch.long, device=X.device).unsqueeze(1)
		medoids = torch.arange(K, device=distance_matrix.device, dtype=torch.long)
		medoids = medoids.unsqueeze(0).repeat(X.shape[0], 1)
		_, medoids[:, 0] = torch.max(l2_norm, dim=1)
		for i in range(1, K):
			sub_dis_matrix = distance_matrix[batch_i, medoids[:, :i], :]			# [B, i, N]
			values, indices = torch.min(sub_dis_matrix, dim=1)						# [B, N]
			values_, indices_ = torch.max(values, dim=1)							# [B]
			medoids[:, i] = indices_

		return medoids

@torch.no_grad()
@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def batch_fast_kmedoids_with_split(X, K, distance='euclidean', threshold=1e-5, iter_limit=60,
									id_sort=True, norm_p=2.0, split_size=4, pre_norm=False):
	"""
	split a batch tensor into multiple chunks in order to avoid OOM of GPU mem
	Args:
		pre_norm: if true, do l2 normalization first before clustering
	"""
	if pre_norm:
		X = X / (X.norm(dim=-1, keepdim=True) + 1e-6)

	if X.shape[0] > split_size:
		all_t = torch.split(X, split_size, dim=0)
		assign_l, medoids_l = [], []
		for x_tmp in all_t:
			assign, medoids = batch_fast_kmedoids(x_tmp, K, distance=distance, threshold=threshold,
													iter_limit=iter_limit,
													id_sort=id_sort, norm_p=norm_p)
			assign_l.append(assign)
			medoids_l.append(medoids)

		return torch.cat(assign_l, dim=0), torch.cat(medoids_l, dim=0)

	else:
		assign, medoids = batch_fast_kmedoids(X, K, distance=distance, threshold=threshold,
												iter_limit=iter_limit,
												id_sort=id_sort, norm_p=norm_p)		
		return assign, medoids


@torch.no_grad()
@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def batch_fast_kmedoids(X, K, distance='euclidean', threshold=1e-5, iter_limit=60,
						id_sort=True, norm_p=2.0):
	"""
	perform batch k mediods
	Args:
		X: (torch.tensor) matrixm, dtype should be torch.float
		K: (int) number of clusters
		distance_matrix: torch.Tensor, pairwise distance matrix of input
		threshold: (float) threshold [default: 0.0001]
		iter_limit: hard limit for max number of iterations
		id_sort: whether sort id of cluster centers in ascending order 
		norm_p: the norm of distance metric
	Return:
		(cluster_assginment, mediods)
	"""		
	assert distance in ['euclidean', 'cosine'] and X.ndim == 3

	B, N, L = X.shape[0], X.shape[1], X.shape[2]
	distance_matrix = pairwise_distance(X, X, metric=distance, all_negative=True,
											self_nearest=True, p=norm_p)
	repeat_dis_m = distance_matrix.unsqueeze(1).repeat(1, K, 1, 1)							# [B, K, N, N]
	# step 1: initialize medoids (KKZ)						
	mediods = KKZ_init(X, distance_matrix, K, batch=True)   								# [B, K]
	batch_i = torch.arange(X.shape[0], dtype=torch.long, device=X.device).unsqueeze(1)		# [B, 1]
	# [B, K, 1]
	K_index = torch.arange(K, dtype=torch.long, device=X.device).reshape(1, K, 1).repeat(B, 1, 1)			

	for step in range(iter_limit):
		# step 2: assign points to medoids
		pre_mediods = mediods
		sub_dis_matrix = distance_matrix[batch_i, mediods, :]								# [B, K, N]
		min_dis, cluster_assginment = torch.min(sub_dis_matrix, dim=1)						# [B, N]

		# step 3: compute medoids
		cluster_assgin_r = cluster_assginment.unsqueeze(1).repeat(1, K, 1)					# [B, K, N]
		mask = (cluster_assgin_r == K_index)												# [B, K, N]
		sub_matrix = repeat_dis_m * mask.unsqueeze(-1) * mask.unsqueeze(-2)					# [B, K, N, N]
		mediods = torch.argmin(torch.sum(sub_matrix, dim=-1), dim=-1)						# [B, K]

		# the shift of mediods
		center_shift = torch.sum((X[batch_i, mediods, :] - X[batch_i, pre_mediods, :]) ** 2,
									dim=-1).sqrt().sum(dim=-1).mean()
		if center_shift < threshold:
			break

	if id_sort:
		mediods, _ = torch.sort(mediods, dim=1)
		# step 2: assign points to medoids
		sub_dis_matrix = distance_matrix[batch_i, mediods, :]								# [B, K, N]
		min_dis, cluster_assginment = torch.min(sub_dis_matrix, dim=1)						# [B, N]

	# print('The step is {}'.format(step))
	return cluster_assginment, mediods


@torch.no_grad()
def fast_kmedoids(X, K, distance_matrix=None, threshold=1e-5, iter_limit=50,
					metric='euclidean', id_sort=True, norm_p=2.0):
	"""
	perform k mediods
	Args:
		X: (torch.tensor) matrixm, dtype should be torch.float
		K: (int) number of clusters
		distance_matrix: torch.Tensor, pairwise distance matrix of input
		threshold: (float) threshold [default: 0.0001]
		iter_limit: hard limit for max number of iterations
	Return:
		(cluster_assginment, mediods)
	"""
	assert X.ndim == 2
	N, D = X.shape
	if distance_matrix is None:
		distance_matrix = pairwise_distance(X, X, metric=metric, all_negative=True,
												self_nearest=True, p=norm_p)
	repeat_dis_m = distance_matrix.unsqueeze(0).repeat(K, 1, 1)						# [K, N, N]
	# step 1: initialize medoids (kmeans++)						
	mediods = KKZ_init(X, distance_matrix, K)                               		# [K]
	K_index = torch.arange(K, dtype=torch.long, device=X.device).unsqueeze(-1)		# [K, 1]

	for step in range(iter_limit):
		# step 2: assign points to medoids
		pre_mediods = mediods
		sub_dis_matrix = distance_matrix[:, mediods]								# [N, K]
		min_dis, cluster_assginment = torch.min(sub_dis_matrix, dim=-1)				# [N]

		# step 3: compute medoids
		cluster_assgin_r = cluster_assginment.unsqueeze(0).repeat(K, 1)				# [K, N]
		mask = (cluster_assgin_r == K_index)										# [K, N]
		sub_matrix = repeat_dis_m * mask.unsqueeze(-1) * mask.unsqueeze(-2)
		mediods = torch.argmin(torch.sum(sub_matrix, dim=-1), dim=-1)				# [K]

		center_shift = torch.sum(torch.sqrt(torch.sum((X[mediods, :] - X[pre_mediods, :]) ** 2, dim=-1)))
		if center_shift < threshold:
			break	

	if id_sort:
		mediods, _ = torch.sort(mediods)
		# step 2: assign points to medoids
		sub_dis_matrix = distance_matrix[:, mediods]								# [N, K]
		min_dis, cluster_assginment = torch.min(sub_dis_matrix, dim=-1)				# [N]

	return cluster_assginment, mediods


if __name__ == "__main__":
	pass
