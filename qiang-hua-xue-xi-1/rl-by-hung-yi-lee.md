---
description: 李老师的强化学习课程笔记
---

# RL by Hung-yi Lee

### 基础知识

#### Difficulties of Reinforcement Learning

1. Reward Delay
2. Agent's action affect the subsequent data it receives

#### **Learning Approches**

1. Policy-based: Learning an Actor
2. Value-based: Leanring a Critic
3. Actor-Critic

EG: AlphaGo: Supervised Learning + Reinforcement Learning

### Learning a Actor

![](https://raw.githubusercontent.com/wang-piaoliang/gitbookimagerepo/master/haohaoxuexi/image-20200921170011430.png)

#### 1. NN As Actor

![](https://raw.githubusercontent.com/wang-piaoliang/gitbookimagerepo/master/haohaoxuexi/image-20200921170045645.png)

![](https://raw.githubusercontent.com/wang-piaoliang/gitbookimagerepo/master/haohaoxuexi/image-20200921170410646.png)

#### 2. Goodness of Actor

![](https://raw.githubusercontent.com/wang-piaoliang/gitbookimagerepo/master/haohaoxuexi/image-20200921170600541.png)

![](https://raw.githubusercontent.com/wang-piaoliang/gitbookimagerepo/master/haohaoxuexi/image-20200921170640392.png)

#### 3. Pick the best function

![](https://raw.githubusercontent.com/wang-piaoliang/gitbookimagerepo/master/haohaoxuexi/image-20200921170723046.png)

### Learning a Critic

![](https://raw.githubusercontent.com/wang-piaoliang/gitbookimagerepo/master/haohaoxuexi/image-20200921170749675.png)

