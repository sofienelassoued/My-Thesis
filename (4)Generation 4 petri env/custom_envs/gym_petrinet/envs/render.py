# Import liberies


import os
import sys
import pygame 
from graphviz import Digraph


#%% Create_Snapshot

sys.path.append(os.getcwd()+"\custom_envs\gym_petrinet\envs")




def graph_generater(places,transitions,action,fired,inprocess):
                     
               g = Digraph('output', format='jpg' ) 
               
               for n in places:     
                   place=str(str(n.name)+" ("+str(len(n.token_list))+")")
                   
                   if n.name in inprocess:
                       g.node(place, color='blue')
                   else: g.node(place, color='black')
                          
               for n in transitions:    
                          
                   if n==action :
                      g.node(str(n),shape="box",color='red')
                   else:g.node(str(n),shape="box",color='black')
              
               
               for i in places:           
                   place=str(str(i.name)+" ("+str(len(i.token_list))+")")
                        
                   for j in i.In_arcs:                 
                       if j==action and fired==True :
                          g.edge(j,place,color='red' )                 
                       else :g.edge(j,place,color='black')
                                          
                   for k in i.Out_arcs :    
                       g.edge(place,k)                        
                       
               return g


def snapshot_creator(graph,simulation_clock,reward,episode_reward,firing,episode=0):
    

        
    white= (255, 255, 255)
    black = (0, 0, 0)
    blue = (0, 0, 255)
              
    pygame.font.init()
    font = pygame.font.Font('freesansbold.ttf', 12)
    font2 = pygame.font.SysFont('arial', 11)
              
 
    graph.render(str(simulation_clock),cleanup=True)         
    image=pygame.image.load(str(simulation_clock)+".jpg") 
    Episode=font.render(str("Episode : "+str (episode)), True, black)   
    Step=font.render(str("Step : "+str (simulation_clock)), True, black)        
    step_Reward=font.render(str("Step Reward : "+str (reward)), True, blue)
    ep_Reward=font.render(str("Episode Reward : "+str (episode_reward)), True, blue)
    firing=font2.render(str(firing), True, blue)
              
    display_width = image.get_width()
    display_height =image.get_height()
              
    if  display_width >700 or display_height :
        display_width=600
        display_height=500
        image = pygame.transform.scale(image, (display_width, display_height))
                  
        screen_shot=pygame.Surface((display_width,display_height+100)) 
        screen_shot.fill(white)
        screen_shot.blits(blit_sequence=((Episode,(5,0)),(Step,(5,20)),(firing,(5,40)),(step_Reward,(5,60)),(ep_Reward,(5,80)),(image,(0,100))))
   
        os.remove(str(simulation_clock)+".jpg")  
        
        return screen_shot

def display(speed,grafic_container,saved_render,replay=False,continues=True):  
          
          position=(0,0)
          white= (255, 255, 255)
          clock = pygame.time.Clock() 
          
          try:
           display_width = (grafic_container[0].get_width())
           display_height =(grafic_container[0].get_height())+100
              
          except:
              display_width=300
              display_height=500
              
          pygame.init()
          pygame.display.init()   
          pygame.display.set_caption('Petrinet')
          Display = pygame.display.set_mode((display_width,display_height))
          Display.fill(white)
           
             
          frame =0  
          clock.tick(1)
          restart=True
          paused=False
               
          if continues==True:
              while True :          
                  if restart==True:
                      
                      for i in range (len(grafic_container)):
                  
                          pygame.time.wait(speed)               
                          if replay==False:Display.blit(grafic_container[i],position)
                          else:Display.blit(saved_render[i],position)  
                          
                          pygame.display.update()
                          for event in pygame.event.get() :
                              if event.type == pygame.QUIT :  
                                  pygame.quit()
                                  break
                              #Pause the animation  use "p" to pause and "c" to continue 
                              elif event.type == pygame.KEYDOWN:
                                  if event.key == pygame.K_p:
                                      paused=True
                                      while paused==True :
                                          Display.blit(saved_render[i],position)
                                          for event in pygame.event.get() :
                                              if event.type == pygame.KEYDOWN:
                                                  if event.key == pygame.K_c:
                                                      paused=False
                                                      break
                      restart=False
                  #replay the animation     
                  for event in pygame.event.get() :
                       if event.type == pygame.QUIT :
                                  pygame.quit()
                                  break
                       elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            restart=True
                            
          else : #play the animation frame by frame use Left and right keys:
          
              while True: 
                
                  if replay==False:Display.blit(grafic_container[frame],position)
                  else:Display.blit(saved_render[frame],position) 
                  pygame.display.update()
                  
                  for event in pygame.event.get() :     
                      if event.type == pygame.QUIT :
                          pygame.quit()
                          break
                     
                      elif event.type == pygame.KEYDOWN:                   
                          if event.key == pygame.K_RIGHT: 
                             if frame <len(grafic_container)-1:frame+=1           
                          elif event.key ==pygame.K_LEFT: 
                              if frame >0 :frame-=1  
    
