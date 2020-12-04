from sklearn.metrics import mean_squared_error
import numpy as np
from utils.Preprocessing import Scale
import math
def RMSE(xPredict, yPredict, xTrue, yTrue, rawValue = False):
    Rx = mean_squared_error(xTrue, xPredict)
    Ry = mean_squared_error(yTrue, yPredict)
    return math.sqrt(Rx + Ry)
def normalization(dataset_obsv, dataset_pred):
    # ================ Normalization ================
    scale = Scale()
    scale.max_x = max(np.max(dataset_obsv[:, :, 0]), np.max(dataset_pred[:, :, 0]))
    scale.min_x = min(np.min(dataset_obsv[:, :, 0]), np.min(dataset_pred[:, :, 0]))
    scale.max_y = max(np.max(dataset_obsv[:, :, 1]), np.max(dataset_pred[:, :, 1]))
    scale.min_y = min(np.min(dataset_obsv[:, :, 1]), np.min(dataset_pred[:, :, 1]))
    scale.calc_scale(keep_ratio=True)
    dataset_obsv = scale.normalize(dataset_obsv)
    dataset_pred = scale.normalize(dataset_pred)
    return dataset_obsv, dataset_pred
# def sample_gaussian_2d(mux, muy, sx, sy, corr, nodesPresent, look_up):
#     '''
#     Parameters
#     ==========
#     mux, muy, sx, sy, corr : a tensor of shape 1 x numNodes
#     Contains x-means, y-means, x-stds, y-stds and correlation
#     nodesPresent : a list of nodeIDs present in the frame
#     look_up : lookup table for determining which ped is in which array index
#     Returns
#     =======
#     next_x, next_y : a tensor of shape numNodes
#     Contains sampled values from the 2D gaussian
#     '''
#     o_mux, o_muy, o_sx, o_sy, o_corr = mux[0, :], muy[0, :], sx[0, :], sy[0, :], corr[0, :]
#
#     numNodes = mux.size()[1]
#     next_x = torch.zeros(numNodes)
#     next_y = torch.zeros(numNodes)
#     converted_node_present = [look_up[node] for node in nodesPresent]
#     for node in range(numNodes):
#         if node not in converted_node_present:
#             continue
#         mean = [o_mux[node], o_muy[node]]
#         cov = [[o_sx[node]*o_sx[node], o_corr[node]*o_sx[node]*o_sy[node]],
#                 [o_corr[node]*o_sx[node]*o_sy[node], o_sy[node]*o_sy[node]]]
#
#         mean = np.array(mean, dtype='float') #1x2
#         cov = np.array(cov, dtype='float') #2x2
#         next_values = np.random.multivariate_normal(mean, cov, 1)
#         next_x[node] = next_values[0][0]
#         next_y[node] = next_values[0][1]
#
#     return next_x, next_y
#
# def Gaussian2DLikelihood(outputs, targets, nodesPresent, look_up):
#     '''
#     params:
#     outputs : predicted locations
#     targets : true locations
#     assumedNodesPresent : Nodes assumed to be present in each frame in the sequence
#     nodesPresent : True nodes present in each frame in the sequence
#     look_up : lookup table for determining which ped is in which array index
#     '''
#     seq_length = outputs.size()[0]
#     # Extract mean, std devs and correlation
#     mux, muy, sx, sy, corr = getCoef(outputs)
#
#     # Compute factors
#     normx = targets[:, :, 0] - mux
#     normy = targets[:, :, 1] - muy
#     sxsy = sx * sy
#
#     z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy) #loss, 1x nNode
#     negRho = 1 - corr**2
#
#     # Numerator
#     result = torch.exp(-z/(2*negRho))
#     # Normalization factor
#     denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))
#
#     # Final PDF calculation
#     result = result / denom
#
#     # Numerical stability
#     epsilon = 1e-20
#
#     result = -torch.log(torch.clamp(result, min=epsilon))
#
#     loss = 0
#     counter = 0
#
#     for framenum in range(seq_length):
#
#         nodeIDs = nodesPresent[framenum]
#         nodeIDs = [int(nodeID) for nodeID in nodeIDs]
#
#         for nodeID in nodeIDs:
#             nodeID = look_up[nodeID]
#             loss = loss + result[framenum, nodeID]
#             counter = counter + 1
#
#     if counter != 0:
#         return loss / counter
#     else:
#         return loss
#
#
# def sample_validation_data(x_seq, Pedlist, grid, args, net, look_up, num_pedlist, dataloader): #僅有一組seq, 多個node
#     '''
#     The validation sample function
#     params:
#     x_seq: Input positions
#     Pedlist: Peds present in each frame
#     args: arguments
#     net: The model
#     num_pedlist : number of peds in each frame
#     look_up : lookup table for determining which ped is in which array index
#     '''
#     # Number of peds in the sequence
#     numx_seq = len(look_up)
#
#     total_loss = 0
#
#     # Construct variables for hidden and cell states
#     with torch.no_grad():
#         hidden_states = Variable(torch.zeros(numx_seq, net.args.rnn_size))
#         if args.use_cuda:
#             hidden_states = hidden_states.cuda()
#         if not args.gru:
#             cell_states = Variable(torch.zeros(numx_seq, net.args.rnn_size))
#             if args.use_cuda:
#                 cell_states = cell_states.cuda()
#         else:
#             cell_states = None
#
#
#         ret_x_seq = Variable(torch.zeros(args.seq_length, numx_seq, 2))
#
#         # Initialize the return data structure
#         if args.use_cuda:
#             ret_x_seq = ret_x_seq.cuda()
#
#         ret_x_seq[0] = x_seq[0]
#
#         # For the observed part of the trajectory
#         for tstep in range(args.seq_length -1):
#             loss = 0
#             # Do a forward prop
#             out_, hidden_states, cell_states = net(x_seq[tstep].view(1, numx_seq, 2), #一個seq有numx_seq nodes, 每個node有一組x,y
#                                                    [grid[tstep]],
#                                                    hidden_states,
#                                                    cell_states,
#                                                    [Pedlist[tstep]], [num_pedlist[tstep]], dataloader, look_up)
#             # loss_obs = Gaussian2DLikelihood(out_obs, x_seq[tstep+1].view(1, numx_seq, 2), [Pedlist[tstep+1]])
#
#             # Extract the mean, std and corr of the bivariate Gaussian
#             mux, muy, sx, sy, corr = getCoef(out_)
#             # Sample from the bivariate Gaussian
#             next_x, next_y = sample_gaussian_2d(mux.data, muy.data, sx.data, sy.data, corr.data, Pedlist[tstep], look_up)
#             ret_x_seq[tstep + 1, :, 0] = next_x
#             ret_x_seq[tstep + 1, :, 1] = next_y
#             loss = Gaussian2DLikelihood(out_[0].view(1, out_.size()[1], out_.size()[2]),
#                                         x_seq[tstep].view(1, numx_seq, 2),
#                                         [Pedlist[tstep]],
#                                         look_up)
#             total_loss += loss
#
#
#     return ret_x_seq, total_loss / args.seq_length