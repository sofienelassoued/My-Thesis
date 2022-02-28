Actions=[]


MCTS_Actions =Action_sequence(Nodes)
conver_dict1= {0:8,1:9,2:12,3:14,4:11,5:13,6:15,7:10 }#from MCTS to Petri

for i in MCTS_Actions :
    Actions.append(conver_dict1[i])



print (MCTS_Actions)
print (Actions)
print (len (MCTS_Actions))

for i in Actions :
    obs, rewards, dones, info = env.step(i,1) 
    

env.render(replay=False,continues=False)

