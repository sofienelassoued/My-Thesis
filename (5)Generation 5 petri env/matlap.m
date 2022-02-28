

 A = [mp_zero mp_zero mp_zero mp_zero mp_zero mp_zero mp_zero mp_zero; 
     mp_zero 1 mp_zero mp_zero mp_zero mp_zero mp_zero mp_zero;
     mp_zero mp_zero 2 mp_zero mp_zero mp_zero mp_zero mp_zero;
     mp_zero mp_zero mp_zero 3 mp_zero mp_zero mp_zero mp_zero;
     mp_zero mp_zero mp_zero mp_zero mp_zero mp_zero mp_zero mp_zero;
     mp_zero mp_zero mp_zero mp_zero 1 mp_zero mp_zero mp_zero;
     mp_zero mp_zero mp_zero mp_zero mp_zero 2 mp_zero mp_zero;
     mp_zero mp_zero mp_zero mp_zero mp_zero mp_zero 3 mp_zero;]
 
 
 hA = [mp_zero mp_zero mp_zero mp_zero ;
       1 mp_zero mp_zero mp_zero ;
       mp_zero 2 mp_zero mp_zero;
       mp_zero mp_zero 3 mp_zero ;]
 
 
 
 
[C, n] = mp_star(hA)



 B = [1 1 mp_zero mp_zero mp_zero mp_zero mp_zero mp_zero; 
     mp_zero mp_zero 1 mp_zero mp_zero mp_zero mp_zero mp_zero; 
     mp_zero mp_zero mp_zero 1 mp_zero mp_zero mp_zero mp_zero;  
     mp_zero mp_zero mp_zero mp_zero mp_zero mp_zero mp_zero mp_zero; 
     mp_zero mp_zero mp_zero mp_zero 1 mp_zero 1 mp_zero; 
     mp_zero mp_zero mp_zero mp_zero mp_zero mp_zero mp_zero 1; 
     mp_zero mp_zero mp_zero mp_zero mp_zero 1 mp_zero mp_zero; 
     mp_zero mp_zero mp_zero mp_zero mp_zero mp_zero mp_zero mp_zero; ]
  
  hB = [1 1 mp_zero mp_zero ;
       mp_zero mp_zero 1 mp_zero ;
       mp_zero mp_zero mp_zero 1;
       mp_zero mp_zero mp_zero mp_zero ;]
 
 U=[3 ;1; 0 ;0; 0 ;0; 0 ;0]
 hU=[1 ;0; 0 ;0]
 
 mp_multi(hB,hU)
