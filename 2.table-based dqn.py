#!/usr/bin/env python
# coding: utf-8

# In[2]:


import gym
import collections
from tensorboardX import SummaryWriter


# In[10]:


ENV_NAME='FrozenLake-v0'
gamma=0.9
ALPHA=0.2
TEST_EPISODES=20
class Agent:
    
    def __init__(self): 
        '''
            func:1)reset the eviroment
                 2)initialize the table Q(s,a)
                a1 a2 a3 ... am
            s1
            s2
            s3
            s4
        '''
        self.env=gym.make(ENV_NAME)
        self.state=self.env.reset()
        print("env initialization fininshed")
        self.values=collections.defaultdict(float)
        
    def sample_env(self):
        '''
            func:sample a tuple for our train set
            args:none
            return(tuple):a train set sample (old_state,action,reward,new_state)
        '''
        ##random sample an action
        action=self.env.action_space.sample()
        ##record the old state s
        old_state=self.state
        new_state,reward,is_done,_=self.env.step(action)
        ##if the episode is finished reset the state
        self.state=self.env.reset() if is_done else new_state
        return (old_state,action,reward,new_state)
    
    def best_value_and_action(self,state):
        '''
            func:find a best_value&action based on state
            args(env):state
            return:best value and action
            notes: the function will be used twice
                   1) test the policy state->best action
                   2) update Q(s,a) accordding to max a' Q(s',a') 
        '''
        best_value,best_action=None,None
        ##search the action_space a1,a2,...,an
        for action in range(self.env.action_space.n):
            ##the current(s,a)
            action_value=self.values[(state,action)]
            if best_value is None or best_value<action_value:
                best_value=action_value
                best_action=action
        return best_value,best_action
    
    def value_update(self,s,a,r,next_s):
        '''
            func:update the table Q(s,a)
            args:
                s(state):the current state
                a(action):the current action
                r(reward):the current reward
                nest_s(state):next state after the action
        '''
        best_v,_=self.best_value_and_action(next_s)
        new_val=r+gamma*best_v
        old_val=self.values[(s,a)]
        self.values[(s,a)]=old_val*(1-ALPHA)+new_val*ALPHA
    
    def play_episode(self,env):
        '''
            func:to evaluate the policy
            args(env):enviroment
            return(float):the total return based on the current policy
        '''
        total_reward=0.0
        state=env.reset()
        while True:
            ##Get the best action based on the current state
            _,action=self.best_value_and_action(state)
            ##perform the action
            new_state,reward,is_done,_=env.step(action)
            total_reward+=reward
            if is_done:
                break
            state=new_state
        return total_reward


# In[11]:


if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-q-learning")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        s, a, r, next_s = agent.sample_env()
        agent.value_update(s, a, r, next_s)

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward
        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()


# In[ ]:




