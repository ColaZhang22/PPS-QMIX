import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from smac.env import StarCraft2Env
import argparse
from replay_buffer import ReplayBuffer
from pfqmix_smac import QMIX_SMAC
from normalization import Normalization
from torch import device
from aggregation import aggragate,reward_aggragate

class Runner_QMIX_SMAC:
    def __init__(self, args, env_name, number, seed):
        self.args = args
        self.env_name = env_name
        self.args.env_name = env_name
        self.number = number
        self.seed = seed
        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Create env
        self.env = StarCraft2Env(map_name=self.env_name, seed=self.seed)
        self.test_env = StarCraft2Env(map_name=self.env_name,seed=self.seed)
        self.env_info = self.env.get_env_info()
        self.args.N = self.env_info["n_agents"]  # The number of agents
        self.args.obs_dim = self.env_info["obs_shape"]  # The dimensions of an agent's observation space
        self.args.state_dim = self.env_info["state_shape"]  # The dimensions of global state space
        self.args.action_dim = self.env_info["n_actions"]  # The dimensions of an agent's action space
        self.args.episode_limit = self.env_info["episode_limit"]  # Maximum number of steps per episode
        self.calculate_limit=10
        print("number of agents={}".format(self.args.N))
        print("obs_dim={}".format(self.args.obs_dim))
        print("state_dim={}".format(self.args.state_dim))
        print("action_dim={}".format(self.args.action_dim))
        print("episode_limit={}".format(self.args.episode_limit))

        # Create N agents
        self.agent_n = QMIX_SMAC(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        # Create a tensorboard
        # self.writer = SummaryWriter(log_dir='./runs/{}/{}_env_{}_number_{}_seed_{}_VDN'.format(self.args.algorithm, self.args.algorithm, self.env_name, self.number, self.seed))
        self.writer = SummaryWriter(
            log_dir='./runs/QMIX/{}_env_{}_number_{}_seed_{}_QMIX'.format(self.args.algorithm,
                                                                       self.env_name, self.number, self.seed))
        # Use batch
        self.device=device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.epsilon = self.args.epsilon  # Initialize the epsilon
        self.win_rates = []  # Record the win rates
        self.total_steps = 0
        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=1)

    def run(self, ):

        evaluate_num = -1  # Record the number of evaluations
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                print('evaluate_num:',evaluate_num)
                self.reward_weights=self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1

            _, _, episode_steps,_ = self.run_episode_smac()  # Run an episode
            self.total_steps += episode_steps

            if self.replay_buffer.current_size >= self.args.batch_size:
                # print("!!!!!!Start training!!!!!")
                self.agent_n.train(self.replay_buffer, self.total_steps,self.calculate_weight)  # Training
                # if self.agent_n.train_step % 100==0:
                #     print("Test after merge")
                #     self.evaluate_policy()

        # self.evaluate_policy()
        self.env.close()

    def calculate_weight(self, ):

        win_times = 0
        evaluate_reward = 0
        rewards_weights = torch.zeros(self.args.N)
        for _ in range(self.calculate_limit):
            win_tag, episode_reward, _, reward_list = self.run_episode_smac(evaluate=True, merge=False)
            if win_tag:
                win_times += 1
            evaluate_reward += episode_reward
            rewards_weights += reward_list

        win_rate = win_times / self.calculate_limit
        print("-----------------------------calculate_weight-------------------------------------------------")
        print("evaluate reward:", evaluate_reward, ' times:', self.calculate_limit, "reward weights:",
              torch.softmax(rewards_weights / self.calculate_limit, dim=0), "epsilon:", self.epsilon)
        evaluate_reward = evaluate_reward / self.calculate_limit
        self.win_rates.append(win_rate)
        print("total_steps:{} \t win_rate:{} \t evaluate_reward:{}".format(self.total_steps, win_rate, evaluate_reward))
        print("-----------------------------calculate_weight-------------------------------------------------")
        # Save the win rates
        return torch.softmax(rewards_weights /self.calculate_limit, dim=0)

    def evaluate_policy(self,):
        win_times = 0
        evaluate_reward = 0
        rewards_weights= torch.zeros(self.args.N)
        for _ in range(self.args.evaluate_times):
            win_tag, episode_reward, _, reward_list = self.run_episode_smac(evaluate=True,merge=False)
            if win_tag:
                win_times += 1
            evaluate_reward += episode_reward
            rewards_weights += reward_list

        win_rate = win_times / self.args.evaluate_times
        print("evaluate reward:",evaluate_reward,' times:',self.args.evaluate_times,"reward weights:",torch.softmax(rewards_weights/self.args.evaluate_times,dim=0),"epsilon:",self.epsilon)
        evaluate_reward = evaluate_reward / self.args.evaluate_times
        self.win_rates.append(win_rate)
        print("total_steps:{} \t win_rate:{} \t evaluate_reward:{}".format(self.total_steps, win_rate, evaluate_reward))
        self.writer.add_scalar('evaluate_reward_{}'.format(self.env_name),evaluate_reward,global_step=self.total_steps)
        self.writer.add_scalar('win_rate_{}'.format(self.env_name), win_rate, global_step=self.total_steps)
        # Save the win rates
        np.save('./data_train/{}_env_{}_number_{}_seed_{}.npy'.format(self.args.algorithm, self.env_name, self.number, self.seed), np.array(self.win_rates))
        return  torch.softmax(rewards_weights/self.args.evaluate_times,dim=0)

    def run_episode_smac(self, evaluate=False, merge=False):
        if self.args.use_rnn:  # If use RNN, before the beginning of each episode，reset the rnn_hidden of the Q network.
            for i in range(self.args.N):
                self.agent_n.eval_Q_net_N[i].rnn_hidden = None
        last_onehot_a_n = np.zeros((self.args.N, self.args.action_dim))  # Last actions of N agents(one-hot)

        win_tag = False
        episode_reward = 0
        last_onehot_a_n = np.zeros((self.args.N, self.args.action_dim))  # Last actions of N agents(one-hot)
        self.env.reset()
        for episode_step in range(self.args.episode_limit):
            obs_n = self.env.get_obs()  # obs_n.shape=(N,obs_dim)
            s = self.env.get_state()  # s.shape=(state_dim,)
            avail_a_n = self.env.get_avail_actions()  # Get available actions of N agents, avail_a_n.shape=(N,action_dim)
            epsilon = 0 if evaluate else self.epsilon
            a_n = self.agent_n.choose_action(obs_n, last_onehot_a_n, avail_a_n, epsilon)
            last_onehot_a_n = np.eye(self.args.action_dim)[a_n]  # Convert actions to one-hot vectors
            r, done, info = self.env.step(a_n)  # Take a step
            win_tag = True if done and 'battle_won' in info and info['battle_won'] else False
            episode_reward += r

            if not evaluate:
                if self.args.use_reward_norm:
                    r = self.reward_norm(r)
                """"
                    When dead or win or reaching the episode_limit, done will be Ture, we need to distinguish them;
                    dw means dead or win,there is no next state s';
                    but when reaching the max_episode_steps,there is a next state s' actually.
                """

            if done and episode_step + 1 != self.args.episode_limit:
                dw = True
            else:
                dw = False

            # Store the transition
            self.replay_buffer.store_transition(episode_step, obs_n, s, avail_a_n, last_onehot_a_n, a_n, r, dw)
            # Decay the epsilon
            self.epsilon = self.epsilon - self.args.epsilon_decay if self.epsilon - self.args.epsilon_decay > self.args.epsilon_min else self.args.epsilon_min

            if done:
                break



        if not evaluate:
            # An episode is over, store obs_n, s and avail_a_n in the last step
            obs_n = self.env.get_obs()
            s = self.env.get_state()
            avail_a_n = self.env.get_avail_actions()
            self.replay_buffer.store_last_step(episode_step + 1, obs_n, s, avail_a_n)

        # Calculate reward_weight in this epsidoe

        reward_weight = torch.zeros(self.args.N)
        """
        merge means that our method extend action to 
        """
        if merge:
            for i in range(self.args.N):
                reward = 0
                self.env.reset()
                for episode_step in range(self.args.episode_limit):
                    obs_n_c = self.env.get_obs()  # obs_n.shape=(N,obs_dim)
                    avail_a_n_c = self.env.get_avail_actions()  # Get available actions of N agents, avail_a_n.shape=(N,action_dim)
                    epsilon_c = 0
                    a_n_c = self.agent_n.choose_action_from_single_module(obs_n_c, last_onehot_a_n, avail_a_n_c,
                                                                          epsilon_c, i)
                    last_onehot_a_n = np.eye(self.args.action_dim)[a_n_c]  # Convert actions to one-hot vectors
                    r, done, info = self.env.step(a_n_c)  # Take a step
                    reward += r
                    if done:
                        break
                reward_weight[i] = reward

        return win_tag, episode_reward, episode_step + 1,reward_weight


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for QMIX and VDN in SMAC environment")
    parser.add_argument("--max_train_steps", type=int, default=int(2e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=32, help="Evaluate times")
    parser.add_argument("--save_freq", type=int, default=int(1e5), help="Save frequency")

    parser.add_argument("--algorithm", type=str, default="QMIX", help="QMIX or VDN")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon_decay_steps", type=float, default=50000, help="How many steps before the epsilon decays to the minimum")
    parser.add_argument("--epsilon_min", type=float, default=0.05, help="Minimum epsilon")
    parser.add_argument("--buffer_size", type=int, default=5000, help="The capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=96, help="Batch size (the number of episodes)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--qmix_hidden_dim", type=int, default=32, help="The dimension of the hidden layer of the QMIX network")
    parser.add_argument("--hyper_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of the hyper-network")
    parser.add_argument("--hyper_layers_num", type=int, default=1, help="The number of layers of hyper-network")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of RNN")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of MLP")
    parser.add_argument("--use_rnn", type=bool, default=True, help="Whether to use RNN")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Orthogonal initialization")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Gradient clip")
    parser.add_argument("--use_lr_decay", type=bool, default=False, help="use lr decay")
    parser.add_argument("--use_RMS", type=bool, default=False, help="Whether to use RMS,if False, we will use Adam")
    parser.add_argument("--add_last_action", type=bool, default=True, help="Whether to add last actions into the observation")
    parser.add_argument("--add_agent_id", type=bool, default=True, help="Whether to add agent id into the observation")
    parser.add_argument("--use_double_q", type=bool, default=True, help="Whether to use double q-learning")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Whether to use reward normalization")
    parser.add_argument("--use_hard_update", type=bool, default=True, help="Whether to use hard update")
    parser.add_argument("--target_update_freq", type=int, default=200, help="Update frequency of the target network")
    parser.add_argument("--tau", type=int, default=0.005, help="If use soft update")
    parser.add_argument("--env_name", type=str, default='3m', help="map smac use")
    parser.add_argument("--gpu_available",type=bool,default='True',help="train in cpu or gpu")

    args = parser.parse_args()
    args.epsilon_decay = (args.epsilon - args.epsilon_min) / args.epsilon_decay_steps

    env_names = ['MMM2', 'corridor', '8m', '3s_vs_5z',  '2s3z']
    env_index = 0
    runner = Runner_QMIX_SMAC(args, env_name=env_names[env_index], number=1, seed=0)
    runner.run()

