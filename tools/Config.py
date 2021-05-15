# Note that the SL_actor_learning_rate is used for both the SL training and RL training periods, please switch the learning when training in different period

# use this parameters for SL training
'''
SL_first_train = True
#SL_actor_learning_rate = 1e-5
SL_actor_learning_rate = 6e-6
#SL_actor_learning_rate = 1e-6
#SL_actor_learning_rate = 6e-7
#SL_actor_learning_rate = 1e-7
SL_learning_rate_decay_start = 5000
SL_learning_rate_decay_every = 200000
SL_batch_size = 180
SL_max_iters = 300001
SL_save_every = 3000
'''


extend_prob_thre = 0.5  # decide whether it is a extending or stop action
unsuitable_thre = 0.05  # decide whether a certain direction is suitable for extending
direction_thre = 0.1  # random prob of a certain direction will be extended
extend_thre_A = 0.01  # decide the scale of extending
extend_thre_B = 0.9
expand_thre_A = 0.99
expand_thre_B = 1



# use this parameters for RL training
'''
SL_first_train = False
#SL_actor_learning_rate = 1e-5
#SL_actor_learning_rate = 6e-6
#SL_actor_learning_rate = 1e-6
#SL_actor_learning_rate = 6e-7
SL_actor_learning_rate = 1e-7
SL_learning_rate_decay_start = 5000
SL_learning_rate_decay_every = 300000
#SL_batch_size = 90
SL_batch_size = 5
SL_max_iters = 300001
SL_save_every = 3000
'''

Critic_first_train = True
#critic_learning_rate = 1e-6
critic_learning_rate = 5e-7
#critic_learning_rate = 1e-7


action_size = 5
#actor_state_size = 1161  # 256+900+5
#actor_state_size = 2061  # 256+900+900+5
#actor_state_size = 2953  # 2048+900+5
actor_state_size = 3853  # 2048+900+900+5

history_actions_length = 10
max_action_steps = 15
accuracy_thre = 0.5

move_ratio = 0.2



Re_state_size = 3848  # 2048+900+900
#Re_state_size = 3852  # 2048+900+900+4
#Re_state_size = 2060  # 256+900+900+4
Re_first_train = True
Re_learning_rate = 1e-5
Re_batch_size = 90
#Re_batch_size = 5

COCO_path = "/home/smj/DataSet/COCO2014/train2014/"
#COCO_path = "/data/smj_data/DataSet/COCO2014/train2014/"  #127
#COCO_path = "/data1/Data/COCO2014/train2014/" #49
#COCO_path = "/home/smj/create/COCO2014/train2014/" #42

