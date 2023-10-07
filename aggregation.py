import collections
from torch.nn import init
from smac.env import StarCraft2Env
def zero_init(net):
    for i in net.state_dict():
        init.zeros_(net.state_dict()[i])
        # init.zeros_(i.weight)
        # init.zeros_(i.bias)
    return  net

def aggragate(N,eval_Q_net_N,aggragate_model):

    aggragate_model=zero_init(aggragate_model)
    for i in range(N):
        for j in aggragate_model.state_dict():
            aggragate_model.state_dict()[j]+=eval_Q_net_N[i].state_dict()[j]/N

    return aggragate_model

def reward_aggragate(N,eval_Q_net_N,aggragate_model,weights):

    aggragate_model=zero_init(aggragate_model)
    for i in range(N):
        for j in aggragate_model.state_dict():
            aggragate_model.state_dict()[j]+=eval_Q_net_N[i].state_dict()[j]*weights[i]
    return aggragate_model


def personalized_aggragate(N,eval_Q_net_N,aggragate_model,perserve_layer):
    '''
    :param N:
    :param eval_Q_net_N:
    :param aggragate_model:
    :param perserve_layer:
    :return:
    '''
    aggragate_model = zero_init(aggragate_model)
    for i in range(N):
        count = 0
        for j in aggragate_model.state_dict():
            aggragate_model.state_dict()[j]+=eval_Q_net_N[i].state_dict()[j]/N
            count+=1
            if count==perserve_layer:
                break
    return aggragate_model

