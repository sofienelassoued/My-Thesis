# Import liberies


import os
import sys
import pygame 
import graphviz 



#%% Create_Snapshot

sys.path.append(os.getcwd()+"\custom_envs\gym_petrinet\envs")

white= (255, 255, 255)
black = (0, 0, 0)
blue = (0, 0, 255)
              


def Tree_creator(Nodes,simulation_clock=0):
    

    graph =  graphviz.Digraph('structs', filename='step',format='jpg' 
                        , node_attr={'shape': 'record'}  )

    for node in Nodes:  

        combination=str(str(node.value_sum)+'|{<f0>'+str( "ID:"+ str (node.id)) +'|'+ str(node.state)+'|<f1>'+ str(node.marking)+'}|'+str(node.visit_count))     
        graph.node(str(node.id), combination,color='red')
        
        if node.selected==True : graph.node(str(node.id), combination,color='red')
        if node.selected==False :graph.node(str(node.id), combination,color='black')
        
        for child in node.children:           
            graph.edge(str(node.id)+':f1',str(child.id)+':f0') 
            


    graph.render(str(simulation_clock),cleanup=True)
    image=pygame.image.load(str(simulation_clock)+".jpg")  
    
    
    
    display_width = image.get_width()
    display_height =image.get_height()
              
 
    image = pygame.transform.scale(image, (display_width, display_height))         
    screen_shot=pygame.Surface((display_width,display_height)) 
    screen_shot.fill(white)
    screen_shot.blit(image,(0,0))
  
    #os.remove(str(simulation_clock)+".jpg")  

    return screen_shot

 


