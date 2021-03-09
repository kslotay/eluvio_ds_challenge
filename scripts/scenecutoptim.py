import torch
import numpy as np

def sceneCutScore(phi_k):
    cos = torch.nn.CosineSimilarity(dim=0)
    f_fn = lambda c_k, p_k: torch.Tensor([cos(torch.Tensor(c_k), torch.Tensor(p)).tolist() for p in p_k])
    fs_fn = lambda x, p_k: (1/len(p_k))*torch.sum(x)
    ft_fn = lambda x: torch.nn.functional.softmax(torch.max(x), dim=0)
    
    f_sum = 0
    for c in range(len(phi_k)):
        c_k = phi_k[c]
        p_k = [phi_k[p] for p in range(len(phi_k)) if p != c]
        if (len(p_k) > 0):
            f_in = f_fn(c_k, p_k)
            f_sum = f_sum + fs_fn(f_in, p_k) + ft_fn(f_in)

    return f_sum

def getSuperShotSet(movie_dataset, bound_preds, sum_shots=True):
    boundary_idx = np.argwhere(np.round(bound_preds.numpy()) == 1).flatten()
    supershotset = []
    start_idx = 0
    for end_idx in boundary_idx:
        supershot = movie_dataset[start_idx:end_idx]
        
        if sum_shots:
            for s in range(1, len(supershot)):                
                supershot[0] = supershot[0] + supershot[s]
            supershot = supershot[0]
                
        start_idx = end_idx
        supershotset.append(supershot.numpy())
        
    return supershotset

def getMaxSceneSum(supershotset, j, k_upper_bound, memo, scenecutscores):
    if j == 0:
        return 0
    elif memo[j][k_upper_bound] != -1:
        return memo[j][k_upper_bound]
    else:
        max_sum = 0
        for l in range(k_upper_bound+1):
            if (scenecutscores[l+1][len(supershotset)-1] == -1):
                scenecutscores[l+1][len(supershotset)-1] = sceneCutScore(supershotset[l+1:])

            cur_k_sum = getMaxSceneSum(supershotset[:l+1], j-1, l, memo, scenecutscores) + \
                        scenecutscores[l+1][len(supershotset)-1]
        
            if cur_k_sum > max_sum:
                memo[j][l] = cur_k_sum
                max_sum = cur_k_sum
            else:
                memo[j][l] = max_sum
        
        return max_sum

def optimizeSceneCutSet(movie_dataset, bound_preds):
    supershotset = getSuperShotSet(movie_dataset, bound_preds)
    memo = [[-1]*len(supershotset) for i in range(len(supershotset))]
    scenecutscores = [[-1]*len(supershotset) for i in range(len(supershotset))]
    for j in range(50, min(len(supershotset), 400)):
        max_sum = getMaxSceneSum(supershotset, j, 10, memo, scenecutscores)
        print(f'j={j}, k_upper_bound={10}, max_score={max_sum}')
    
    return memo, scenecutscores