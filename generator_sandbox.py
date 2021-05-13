#     def N_generator():
        
#         for i in np.arange(len(N11_vec0)):
#             N10_upper = min((n - N11_vec0[i] - N01_vec0[i]), np.floor(N01_vec0[i] + ntau_obs))
            
#             if N10_vec0[i] <= N10_upper:
#                 for j in np.arange(N10_vec0[i], N10_upper + 1):
#                     N10_val = int(j)
#                     N11_val = int(N11_vec0[i])
#                     N01_val = int(N01_vec0[i])
#                     if check_compatible_2(n11, n10, n01, n00, N11_val, N10_val, N01_val):
#                         yield [N11_val, N10_val, N01_val, n-(N11_val+N10_val+N01_val)]

#     tau_min = math.inf
#     N_accept_min = np.nan
#     tau_max = -math.inf
#     N_accept_max = np.nan
    
#     for tbl in N_generator():
#         tN = (tbl[1] - tbl[2]) / n
#         if tN < tau_min:
#             tau_min = tN
#             N_accept_min = np.array(tbl[:4])
#         if tN > tau_max:
#             tau_max = tN
#             N_accept_max = np.array(tbl[:4])