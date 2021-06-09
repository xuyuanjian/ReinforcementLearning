#!/usr/bin/env python
# coding: utf-8

# In[96]:


import torch
from torch import nn
from collections import namedtuple
import gym
from torch import optim
import numpy as np
HIDDEN_SIZE=128
BATCH_SIZE=16
PERCENTILE=70


# In[97]:


class Net(nn.Module):
    def __init__(self,obs_size,hidden_size,n_actions):
        super(Net,self).__init__()
        self.net=nn.Sequential(
            nn.Linear(
                 in_features=obs_size,
                 out_features=hidden_size,
                 bias=True
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=hidden_size,
                out_features=n_actions,
                bias=True
            )
        )
    def forward(self,x):
        return self.net(x)


# In[98]:


##reward即为total_reward，steps为episode_steps=[]
Episode=namedtuple('Episode',field_names=['reward','steps'])
EpisodeStep=namedtuple('EpisodeStep',field_names=['observation','action','reward'])


# In[99]:


def iterate_batches(env,net,batch_size):
    
    '''
        function:get a episode_batch include a number of episode(batch_size)
        args:
            env(gym.env):enviroment
            net(nn.Module):a net work,input a obs output a vector of actions(probabilty)
            batch_size(int):the amount of episode_steps included in an episode
        return:
            batch(list):a list of episode_step(episdode)
        notes:
            episode_batch:including series of episode
            episode:including series of episode_step
            episode_step:the smallest unit  (observation,step,reward)
    '''
    
    ##output
    batch=[]
    ##accumulate the reward of a episode
    episode_reward=0.0
    ##add the episode_step into an episode until it's finished
    episode_steps=[]
    ##initialize the enviroment
    obs=env.reset() 
    ##transform the output to a vector of probability
    sm=nn.Softmax(dim=1)
    while True:
        ## add a dim for batch_size
        obs_v=torch.FloatTensor([obs])
        ##print("obs_v.shape = %s" %str(obs_v.shape))
        ##transform the output to a vector consisted of probability
        act_probs_v=sm(net(obs_v))
        ##print("act_probs_v.shape = %s" %str(act_probs_v.shape))
        ##unzip the dim batch size to form a list of probability
        act_probs=act_probs_v.data.numpy()[0]
        ##print("act_probs.shape = %s"%str(act_probs.shape))
        ##random choice an action based on the probability
        action=np.random.choice(len(act_probs),p=act_probs)
        ##take the action
        next_obs,reward,is_done,_=env.step(action)
        ## accumulate the reward of an episode
        episode_reward+=reward
        ##add an episode_step to an episode
        episode_steps.append(EpisodeStep(observation=obs,action=action,reward=reward))
        ##if this episode is finished
        if is_done:
            ##add the episode to an episode_batch
            batch.append(Episode(reward=episode_reward,steps=episode_steps))
            ##make the reward of an episode to zero
            episode_reward=0.0
            ##clear the list of an episdoe
            episode_steps=[]
            ##reset the enviroment
            next_obs=env.reset()
            ##if the number of episode in an episode_batch is enough then return
            if len(batch)==batch_size:
                yield batch
                batch=[]
        ## if this episode isn't finished ,continue
        obs=next_obs


# In[105]:


def filter_batch(batch,percentile):
    '''
        function:
        args:
        return:
        notes:
    '''
    ##lamba表达式：输入一个episode获取其中的reward。map表达式：作用于所有的batch
    rewards=list(map(lambda s:s.reward,batch))
    reward_bound=np.percentile(rewards,percentile)
    reward_mean=np.mean(rewards)   
    train_obs=[]
    train_act=[]
    for example in batch:
        if example.reward<reward_bound:
            continue
        train_obs.extend(map(lambda step:step.observation,example.steps))
        train_act.extend(map(lambda step:step.action,example.steps))
        ##print("train_obs.shape:%d"%len(train_obs))
        ##print("train_act.shape:%d"%len(train_act))
    train_obs_v=torch.FloatTensor(train_obs)
    print("train_obs_v.shape:%s"%str(train_obs_v.shape))
    train_act_v=torch.LongTensor(train_act)
    print("train_act_v.shape:%s"%str(train_act_v.shape))
    return train_obs_v,train_act_v,reward_bound,reward_mean


# In[106]:


env=gym.make("CartPole-v0")
obs_size=env.observation_space.shape[0]
n_actions=env.action_space.n
net=Net(obs_size,HIDDEN_SIZE,n_actions)
type(env)


# In[107]:


print(net)


# In[108]:


loss_function=nn.CrossEntropyLoss()
optimizer=optim.Adam(params=net.parameters(),lr=0.01)


# In[109]:


for iter_no,batch in enumerate(iterate_batches(env,net,BATCH_SIZE)):
    obs_v,acts_v,reward_b,reward_m=filter_batch(batch,PERCENTILE)
    optimizer.zero_grad()
    action_scores_v=net(obs_v)
    loss_v=loss_function(action_scores_v,acts_v)
    loss_v.backward()
    optimizer.step()
    print("%d:loss=%.3f,reward_mean=%.1f,reward_bound=%.1f"%(iter_no,loss_v.item(),reward_m,reward_b))


# In[ ]:




