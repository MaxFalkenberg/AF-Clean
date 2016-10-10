# AF-Clean
Master's project modelling the spontaneous emergence of Atrial Fibrillation in a simple Cellular Automata based on work by Kim Christensen et al in PRL 114, 028104 (2015).


Example of how to define tissue and propagate signal:
```python
a = Heart(nu = 0.1,delta = 0.05,eps = 0.05,rp = 50)
```
1. Without periodic pulsing

  * Default pacemaker cells: 
  ```python
  SKIP
  ```
  
  * Custom pacemaker cells:
  ```python
  a.set_pulse(rate = 0, vectors = [[y1,y2,y3...],[x1,x2,x3...]])  

  ```
2. With periodic pulsing at rate P

 * Default pacemaker cells:
  ```python
  a.set_pulse(rate = P) THEN a.pulse()
  ``` 
  * Custom pacemaker cells:
  ```python
  a.set_pulse(rate = P, vectors = [[y1,y2,y3...],[x1,x2,x3...]])
  ```
```
Propagate excitations for T time steps:
 ```python
 a.propagate(t_steps = T)
 ```
  
Save list of excited cells from each time step:
  
  ```python
  a.save('file_name')
  ```
  

