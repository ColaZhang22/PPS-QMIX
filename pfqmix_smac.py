import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mix_net import QMIX_Net, VDN_Net
from aggregation import aggragate,personalized_aggragate,reward_aggragate

# orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


class Q_network_RNN(nn.Module):
    def __init__(self, args, input_dim):
        super(Q_network_RNN, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.action_dim)
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2)

    def forward(self, inputs):
        # When 'choose_action', inputs.shape(N,input_dim)
        # When 'train', inputs.shape(bach_size*N,input_dim)
        x = F.relu(self.fc1(inputs))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        Q = self.fc2(self.rnn_hidden)
        return Q


class Q_network_MLP(nn.Module):
    def __init__(self, args, input_dim):
        super(Q_network_MLP, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, args.action_dim)
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, inputs):
        # When 'choose_action', inputs.shape(N,input_dim)
        # When 'train', inputs.shape(bach_size,max_episode_len,N,input_dim)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        Q = self.fc3(x)
        return Q


class QMIX_SMAC(object):
    def __init__(self, args):
        self.N = args.N
        self.action_dim = args.action_dim
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.add_last_action = args.add_last_action
        self.add_agent_id = args.add_agent_id
        self.max_train_steps=args.max_train_steps
        self.lr = args.lr
        self.gamma = args.gamma
        self.use_grad_clip = args.use_grad_clip
        self.batch_size = args.batch_size  # 这里的batch_size代表有多少个episode
        self.target_update_freq = args.target_update_freq
        self.tau = args.tau
        self.use_hard_update = args.use_hard_update
        self.use_rnn = args.use_rnn
        self.algorithm = args.algorithm
        self.use_double_q = args.use_double_q
        self.use_RMS = args.use_RMS
        self.use_lr_decay = args.use_lr_decay
        self.env_name=args.env_name
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # peronalized federated learning

        self.eval_Q_net_N = []
        self.target_Q_net_N = []
        # Compute the input dimension
        # self.obs_dim_single_agent = int(self.obs_dim / self.N)
        self.input_dim = self.obs_dim

        if self.add_last_action:
            print("------add last action------")
            self.input_dim += self.action_dim
        if self.add_agent_id:
            print("------add agent id------")
            self.input_dim += self.N

        self.aggragte_model =Q_network_RNN(args, self.input_dim).to(self.device)

        for i in range(args.N):
            if self.use_rnn:
                print("------use RNN------")
                self.eval_Q_net_N.append(Q_network_RNN(args, self.input_dim).to(self.device))
                self.target_Q_net_N.append(Q_network_RNN(args, self.input_dim).to(self.device))
                # self.eval_Q_net = Q_network_RNN(args, self.input_dim)
                # self.target_Q_net = Q_network_RNN(args, self.input_dim)
            else:
                print("------use MLP------")
                self.eval_Q_net_N.append(Q_network_MLP(args, self.input_dim).to(self.device))
                self.target_Q_net_N.append(Q_network_MLP(args, self.input_dim).to(self.device))
                # self.eval_Q_net = Q_network_MLP(args, self.input_dim)
                # self.target_Q_net = Q_network_MLP(args, self.input_dim)
            self.target_Q_net_N[i].load_state_dict(self.eval_Q_net_N[i].state_dict())

        if self.algorithm == "QMIX":
            print("------algorithm: QMIX------")
            self.eval_mix_net = QMIX_Net(args).to(self.device)
            self.target_mix_net = QMIX_Net(args).to(self.device)

            if self.use_RMS:
                self.optimizer = [torch.optim.RMSprop(list(self.eval_mix_net.parameters()), lr=self.lr)]
            else:
                self.optimizer = [torch.optim.Adam(list(self.eval_mix_net.parameters()), lr=self.lr)]

        elif self.algorithm == "VDN":
            print("------algorithm: VDN------")
            self.eval_mix_net = VDN_Net().to(self.device)
            self.target_mix_net = VDN_Net().to(self.device)
            self.optimizer=[]
        else:
            print("wrong!!!")
        self.target_mix_net.load_state_dict(self.eval_mix_net.state_dict())

        self.eval_parameters = list(self.eval_mix_net.parameters())
        for i in reversed(range(args.N)):
            self.eval_parameters += list(self.eval_Q_net_N[i].parameters())

        if self.use_RMS:
            print("------optimizer: RMSprop------")
            for i in range(args.N):
                self.optimizer.append(torch.optim.RMSprop(list(self.eval_Q_net_N[i].parameters()),lr=self.lr))
        else:
            # self.optimizer = [torch.optim.Adam(list(self.eval_mix_net.parameters()), lr=self.lr)]
            print("------optimizer: Adam------")
            for i in range(args.N):
                self.optimizer.append(torch.optim.Adam(list(self.eval_Q_net_N[i].parameters()), lr=self.lr))

        # if self.use_RMS:
        #     print("------optimizer: RMSprop------")
        #     self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=self.lr)
        # else:
        #     print("------optimizer: Adam------")
        #     self.optimizer = torch.optim.Adam(self.eval_parameters, lr=self.lr)

        self.train_step = 0

    def choose_action(self, obs_n, last_onehot_a_n, avail_a_n, epsilon):
        with torch.no_grad():
            obs_n = torch.tensor(obs_n, dtype=torch.float32)  # obs_n.shape=(N，obs_dim)
            obs_n_itr = torch.chunk(obs_n, chunks=self.N, dim=0)

            if self.add_last_action:
                last_a_n = torch.tensor(last_onehot_a_n, dtype=torch.float32)
            last_a_n_itr = torch.chunk(last_a_n, chunks=self.N, dim=0)

            if self.add_agent_id:
                agent_id = torch.eye(self.N)
            agent_id_itr =  torch.chunk(agent_id, chunks=self.N, dim=0)

            avail_a_n = torch.tensor(avail_a_n, dtype=torch.float32)  # avail_a_n.shape=(N, action_dim)
            avail_a_n_itr=torch.chunk(avail_a_n, chunks=self.N, dim=0)

            a_n=[]

            #start to collect trajectory
            if np.random.uniform() < epsilon:  # epsilon-greedy
                # Only available actions can be chosen
                a_n = [np.random.choice(np.nonzero(avail_a).flatten()) for avail_a in avail_a_n]
            else:
                for i in range(self.N):
                    inputs = []
                    inputs.append(obs_n_itr[i])
                    if self.add_last_action:
                        inputs.append(last_a_n_itr[i])
                    if self.add_agent_id:
                        inputs.append(agent_id_itr[i])
                    inputs = torch.cat([x for x in inputs], dim=-1)  # inputs.shape=(N,inputs_dim)
                    inputs = inputs.to(self.device)
                    q_value = self.eval_Q_net_N[i](inputs)
                    q_value[avail_a_n_itr[i] == 0] = -float('inf')  # Mask the unavailable actions
                    a_n.append(q_value.cpu().argmax(dim=-1).numpy())
        a_n=np.array(a_n).ravel()
        return a_n

    def choose_action_from_single_module(self, obs_n, last_onehot_a_n, avail_a_n, epsilon,i):
        with torch.no_grad():
            if np.random.uniform() < epsilon:  # epsilon-greedy
                # Only available actions can be chosen
                a_n = [np.random.choice(np.nonzero(avail_a)[0]) for avail_a in avail_a_n]
            else:
                inputs = []
                obs_n = torch.tensor(obs_n, dtype=torch.float32)  # obs_n.shape=(N，obs_dim)
                inputs.append(obs_n)
                if self.add_last_action:
                    last_a_n = torch.tensor(last_onehot_a_n, dtype=torch.float32)
                    inputs.append(last_a_n)
                if self.add_agent_id:
                    inputs.append(torch.eye(self.N))

                inputs = torch.cat([x for x in inputs], dim=-1)  # inputs.shape=(N,inputs_dim)
                inputs = inputs.to(self.device)

                q_values=[]
                for input in inputs:
                    input=input.reshape(1,-1)
                    q_value = self.eval_Q_net_N[i](input)
                    q_values.append(q_value)

                q_values =  torch.cat(q_values)
                avail_a_n = torch.tensor(avail_a_n, dtype=torch.float32)  # avail_a_n.shape=(N, action_dim)
                q_values[avail_a_n == 0] = -float('inf')  # Mask the unavailable actions
                a_n = q_values.argmax(dim=-1).numpy()

            return a_n

    def train(self, replay_buffer, total_steps,calculate_reward_weight):
        #
        # if self.train_step % 300 == 0:
        #     print("AFL")
        #     print("!!!!!Before aggregate---result!!!!!")
        #     reward_weights = calculate_reward_weight()
        #     reward_weights = reward_weights.to(self.device)
        #     # aggragate_model = reward_aggragate(self.N, self.eval_Q_net_N, self.aggragte_model, reward_weights)
        #     aggragate_model = aggragate(self.N, self.eval_Q_net_N, self.aggragte_model)
        #     aggragate_model = aggragate_model.to(self.device)
        #     for i in range(self.N):
        #         self.eval_Q_net_N[i].load_state_dict(aggragate_model.state_dict())
        #     print('Finish aggregating model: round:',self.train_step)
        #     print("!!!!!After aggregate---result!!!!!")
        #     # reward_weights = calculate_reward_weight()

        batch, max_episode_len = replay_buffer.sample()  # Get training data
        self.train_step += 1
        inputs = self.get_inputs(batch, max_episode_len)  # inputs.shape=(bach_size,max_episode_len+1,N,input_dim)
        inputs = inputs.to(self.device)
        # #
        if self.train_step % 300==0:
            print("PFL aggregation")
            layer=4
            aggragate_model = personalized_aggragate(self.N, self.eval_Q_net_N, self.aggragte_model, layer)
            aggragate_model = aggragate_model.to(self.device)
            aggragate_model_weight = aggragate_model.state_dict()

            for i in range(self.N):
                count=0
                personal_weight =  self.eval_Q_net_N[i].state_dict()
                for j in personal_weight:
                    personal_weight[j]=aggragate_model_weight[j]
                    count += 1
                    if count == layer:
                        break
                self.eval_Q_net_N[i].load_state_dict(personal_weight)
            print('Finish aggregating model: round:',self.train_step)

        if self.use_rnn:
            q_evals, q_targets= [], []
            #train for each agent
            for i in range(self.N):
                    for k in range(self.N):
                        self.eval_Q_net_N[k].rnn_hidden = None
                        self.target_Q_net_N[k].rnn_hidden = None
                    q_evals_single, q_targets_single = [], []
                    for t in range(max_episode_len):  # t=0,1,2,...(episode_len-1)
                        #single = inputs[:, t, i].reshape(-1, self.input_dim)
                        q_eval = self.eval_Q_net_N[i](inputs[:, t, i].reshape(-1, self.input_dim))  # q_eval.shape=(batch_size*N,action_dim)
                        q_target = self.target_Q_net_N[i](inputs[:, t + 1, i].reshape(-1, self.input_dim))
                        q_evals_single.append(q_eval.reshape(self.batch_size, 1, -1))  # q_eval.shape=(batch_size,1,action_dim)
                        q_targets_single.append(q_target.reshape(self.batch_size, 1, -1))  # q_eval_single.shape=(batch_size,max_episode_len,1,action_dim)
                    # Stack them according to the time (dim=1)
                    q_evals.append(torch.stack(q_evals_single, dim=1))  # q_evals.shape=(batch_size,max_episode_len,N,action_dim)
                    q_targets.append(torch.stack(q_targets_single, dim=1))
            q_evals=torch.cat(q_evals,dim=2)
            q_targets=torch.cat(q_targets,dim=2)
        else:
        # This part still need to be changed.
            for i in range(self.N):
                q_evals = self.eval_Q_net_N[i](inputs[:, :-1])  # q_evals.shape=(batch_size,max_episode_len,N,action_dim)
                q_targets = self.target_Q_net_N[i](inputs[:, 1:])

        with torch.no_grad():
            if self.use_double_q:  # If use double q-learning, we use eval_net to choose actions,and use target_net to compute q_target
                q_eval_last_single=[]
                for i in range(self.N):
                    a=inputs[:, -1, i].reshape(-1, self.input_dim)
                    q_eval_last_single.append(self.eval_Q_net_N[i](inputs[:, -1,  i].reshape(-1, self.input_dim)).reshape(self.batch_size, 1, 1, -1))
                q_eval_last=torch.cat(q_eval_last_single,dim=2)
                q_evals_next = torch.cat([q_evals[:, 1:], q_eval_last], dim=1) # q_evals_next.shape=(batch_size,max_episode_len,N,action_dim)
                q_evals_next[batch['avail_a_n'][:, 1:] == 0] = -999999
                a_argmax = torch.argmax(q_evals_next, dim=-1, keepdim=True)  # a_max.shape=(batch_size,max_episode_len, N, 1)
                q_targets = torch.gather(q_targets, dim=-1, index=a_argmax).squeeze(-1)  # q_targets.shape=(batch_size, max_episode_len, N)
            else:
                q_targets[batch['avail_a_n'][:, 1:] == 0] = -999999
                q_targets = q_targets.max(dim=-1)[0]  # q_targets.shape=(batch_size, max_episode_len, N)

        # batch['a_n'].shape(batch_size,max_episode_len, N)
        q_evals = torch.gather(q_evals, dim=-1, index=batch['a_n'].unsqueeze(-1).to(self.device)).squeeze(-1)  # q_evals.shape(batch_size, max_episode_len, N)

        # Compute q_total using QMIX or VDN, q_total.shape=(batch_size, max_episode_len, 1)
        if self.algorithm == "QMIX":
            '''
            batch['s'][:,:-1] means [:,:-1,:], represents all elements except last element in second column-current state
            batch['s'][:,1:] means [:,1:,:], represents all elements except first element in second column-next state
            '''
            q_total_eval = self.eval_mix_net(q_evals, batch['s'][:, :-1].to(self.device))
            q_total_target = self.target_mix_net(q_targets, batch['s'][:, 1:].to(self.device))
        else:
            q_total_eval = self.eval_mix_net(q_evals)
            q_total_target = self.target_mix_net(q_targets)
        # targets.shape=(batch_size,max_episode_len,1)

        targets = batch['r'].to(self.device) + self.gamma * (1 - batch['dw'].to(self.device)) * q_total_target


        td_error = (q_total_eval - targets.detach())
        mask_td_error = td_error * batch['active'].to(self.device)
        loss = (mask_td_error ** 2).sum() / (batch['active'].sum().to(self.device))

        for i in range(len(self.optimizer)):
            self.optimizer[i].zero_grad()

        loss.backward()

        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.eval_parameters, 10)
        for i in range(len(self.optimizer)):
            self.optimizer[i].step()


        if self.use_hard_update:
            # hard update
            if self.train_step % self.target_update_freq == 0:
                for i in range(self.N):
                    self.target_Q_net_N[i].load_state_dict(self.eval_Q_net_N[i].state_dict())
                self.target_mix_net.load_state_dict(self.eval_mix_net.state_dict())
                # print("Finish update network")
        else:
            # Softly update the target networks
            for i in range(self.N):
                for param, target_param in zip(self.eval_Q_net_N[i].parameters(), self.target_Q_net_N[i].parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.eval_mix_net.parameters(), self.target_mix_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if self.use_lr_decay:
            self.lr_decay(total_steps)


        #     print('Finish aggregating model: round:', self.train_step)
            # for name, param in aggragate_model.named_parameters():
            #     print(name, param.data)
            #     break

    def lr_decay(self, total_steps):  # Learning rate Decay
        lr_now = self.lr * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer.param_groups:
            p['lr'] = lr_now

    def get_inputs(self, batch, max_episode_len):
        inputs = []
        inputs.append(batch['obs_n'])
        if self.add_last_action:
            inputs.append(batch['last_onehot_a_n'])
        if self.add_agent_id:
            agent_id_one_hot = torch.eye(self.N).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, max_episode_len + 1, 1, 1)
            inputs.append(agent_id_one_hot)

        # inputs.shape=(bach_size,max_episode_len+1,N,input_dim)
        inputs = torch.cat([x for x in inputs], dim=-1)
        return inputs

    def save_model(self, env_name, algorithm, number, seed, total_steps):
        for i in range(self.N):
            torch.save(self.eval_Q_net_N[i].state_dict(), "./model/{}/{}_eval_rnn_number_{}_seed_{}_step_{}k.pth".format(env_name, algorithm, number, seed, int(total_steps / 1000)))
