---
description: 李老师的强化学习课程笔记
---

# RL by Hung-yi Lee

## Basics

### Difficulties of Reinforcement Learning

1. Reward Delay
2. Agent's action affect the subsequent data it receives

### **Learning Approches**

1. Policy-based: Learning an Actor
2. Value-based: Leanring a Critic
3. Actor-Critic

EG: AlphaGo: Supervised Learning + Reinforcement Learning

## Learning a Actor  

![](../.gitbook/assets/image%20%2826%29.png)

### 1. NN As Actor

![](../.gitbook/assets/image%20%2835%29.png)

![](../.gitbook/assets/image%20%2829%29.png)

### 2. Goodness of Actor

![](../.gitbook/assets/image%20%2824%29.png)

![](../.gitbook/assets/image%20%2827%29.png)

### 3. Pick the best function

**Gradient Ascent**

![](../.gitbook/assets/image%20%2837%29.png)

## Learning a Critic

![](../.gitbook/assets/image%20%2832%29.png)

