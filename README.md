# AF-Clean
Master's project modelling the spontaneous emergence of Atrial Fibrillation in a simple Cellular Automata based on work by Kim Christensen et al in PRL 114, 028104 (2015).


Example of how to define tissue and propagate signal:

a = Heart(nu = 0.1,delta = 0.05,eps = 0.05,rp = 50)

Without periodic pulsing

  Default pacemaker cells: a.pulse()
  
  Custom pacemaker cells: a.set_pulse(rate = 0, vectors = [[y1,y2,y3...],[x1,x2,x3...]])  THEN a.pulse()
 
With periodic pulsing at rate P

  Default pacemaker cells: a.set_pulse(rate = P) THEN a.pulse()
  
  Custom pacemaker cells: a.set_pulse(rate = P, vectors = [[y1,y2,y3...],[x1,x2,x3...]])  THEN a.pulse()
  
Propagate excitations for T time steps
 
 a.propagate(t_steps = T)
  
Save list of excited cells from each time step
  
  a.save('file_name')
  

