import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import csv
import scipy.optimize
import math
import os
import datetime
import sys

file_name =  os.path.basename(sys.argv[0])
file_name = file_name[:-3]
tid = datetime.datetime.now().strftime("%d-%m-%y_%H-%M") 

# Vi laver lister ud fra det eksperimentielle data fra vores csv fil
fil="Kapoor_2018"
mod_data=[]
Al_data=[]
R_data=[]
Q3_data=[]
Q4_data=[]
Q2_data=[]
Q1_data=[]
Q0_data=[]
Al5_data=[]
Al4_data=[]

with open("{}.csv".format(fil), newline='') as csvfile:
    spamreader=csv.reader(csvfile,delimiter=';', quotechar="|")
    for column in spamreader:
        mod_data.append(column[0])
        Al_data.append(column[1])
        R_data.append(column[3])
        Q3_data.append(column[4])
        Q4_data.append(column[5])
        Q2_data.append(column[6])
        Q1_data.append(column[7])
        Q0_data.append(column[8])
        Al5_data.append(column[9])
        Al4_data.append(column[10])

# Vi laver strings om til tal med decimaler
mod_data = [float(i) for i in mod_data[1:13]]
Al_data = [float(i) for i in Al_data[1:13]]
R_data = [float(i) for i in R_data[1:13]]
Q3_data = [float(i) for i in Q3_data[1:13]]
Q4_data = [float(i) for i in Q4_data[1:13]]
Q2_data = [float(i) for i in Q2_data[1:13]]
Q1_data = [float(i) for i in Q1_data[1:13]]
Q0_data = [float(i) for i in Q0_data[1:13]]
Al5_data = [float(i) for i in Al5_data[1:13]]
Al4_data = [float(i) for i in Al4_data[1:13]]

# Fra liste til array (?)
Q3_data = np.array(Q3_data)
Q4_data = np.array(Q4_data)
Q2_data = np.array(Q2_data)
Q1_data = np.array(Q1_data)
Q0_data = np.array(Q0_data)
Al5_data = np.array(Al5_data)
Al4_data = np.array(Al4_data)

# modifier liste

draw_nr = list(range(101))
draw_ar = np.array(draw_nr)

M2O = [ ]

for i in draw_ar:
    next_mod = draw_ar[i] / (100 + draw_ar[i]) * 100
    M2O.append(next_mod)
    
M2O.append(50)

SSEtrue = 1

# Model
def model(w1):
    w=np.array([1, abs(w1[0]), abs(w1[1]), abs(w1[2]), abs(w1[3]), abs(w1[4]), abs(w1[5]), abs(w1[6]), abs(w1[7]), abs(w1[8]), abs(w1[9]), abs(w1[10])])
    iw = 1/w
    
    global Q3_m
    global Q4_m
    global Q2_m
    global Q1_m
    global Q0_m
    global Al5_m
    global Al4_m
    global Q3
    
    
    global Al5Na
    global Al4Na
    global Q4Na 
    global Q3Na
    global Q2Na   
    global Q1Na   
    global Q0Na
    
    global Al_draw


    global Q4A
    global Q2A
    global Q1AA
    global Q0AAA 
    
    global Al5
    global Al4A
    global Q3shinanigens
    
    global Q1AN
    global Q0AAN
    global Q0ANN
    
    
    
    Q3_m = []
    Q4_m = []
    Q2_m = []
    Q1_m = []
    Q0_m = []
    Al5_m = []
    Al4_m = []
    
    
# Startværdier
    for ind, i in enumerate(R_data, start=0):    
    
        R_ind = ind   
        Q3 = [(100/(i+1)), ]
        Al5 = [(100*i/(i+1)),  ]
        
        Al4A = [0,  ]
        Al4N = [0,  ]
        
        #Q4
        Q4A = [0, ]
        Q4N = [0, ]
        
        #Q2
        Q2A = [0, ]
        Q2N = [0, ]
        
        #Q1
        Q1NN = [0, ]
        Q1AN = [0, ]
        Q1AA = [0, ]
        
        #Q0
        Q0AAA = [0, ]
        Q0AAN = [0, ]
        Q0ANN = [0, ]
        Q0NNN = [0, ] 
        
        Al_draw = [0, ]
        
        Q3Na  = []
        Al5Na = []
        Al4Na = []
        Q4Na  = []
        Q2Na  = []
        Q1Na  = []
        Q0Na  = [] 
        
        Q3shinanigens = []
        ### Draw of all Al
        if Al5[-1] > 0: 
            while (Al_draw[-1]/3)*100/(100+(Al_draw[-1]/3)) < Al5[-1]*100/(100-Al4A[-1]):
                gQ3 = (Q3[-1]*w[6])*(1/(R_data[R_ind]+1)) / (Q3[-1]*w[6]+Q4A[-1]*w[7]+Q2A[-1]*w[8]+Q1AA[-1]*w[9]+Al5[-1]*w[10])
                gQ4 = (Q4A[-1]*w[7])*(1/(R_data[R_ind]+1)) / (Q3[-1]*w[6]+Q4A[-1]*w[7]+Q2A[-1]*w[8]+Q1AA[-1]*w[9]+Al5[-1]*w[10])
                gQ2 = (Q2A[-1]*w[8])*(1/(R_data[R_ind]+1)) / (Q3[-1]*w[6]+Q4A[-1]*w[7]+Q2A[-1]*w[8]+Q1AA[-1]*w[9]+Al5[-1]*w[10])
                gQ1 = (Q1AA[-1]*w[9])*(1/(R_data[R_ind]+1)) / (Q3[-1]*w[6]+Q4A[-1]*w[7]+Q2A[-1]*w[8]+Q1AA[-1]*w[9]+Al5[-1]*w[10])
                gAl5 = (Al5[-1]*w[10])*(1/(R_data[R_ind]+1)) / (Q3[-1]*w[6]+Q4A[-1]*w[7]+Q2A[-1]*w[8]+Q1AA[-1]*w[9]+Al5[-1]*w[10])
                
                rgQ3 = gQ3 / (gQ3 + gQ2 + gQ1 + gAl5)
                rgQ2 = gQ2 / (gQ3 + gQ2 + gQ1 + gAl5)
                rgQ1 = gQ1 / (gQ3 + gQ2 + gQ1 + gAl5)
                rgAl5 = gAl5 / (gQ3 + gQ2 + gQ1 + gAl5)
                
                #overgang
                if (Q4A[-1] + Q2A[-1] + 2*Q1AA[-1] + 3*Q0AAA[-1]) < Q3[0]*w[5]:
                    P = 1
                else:
                    P = 0
                
                # Draws        
                if Q3[-1] - gQ3 + (-rgQ3)*gQ4 > 0:
                    next_Q3 = Q3[-1] - gQ3 + (- rgQ3)*gQ4
                    
                else: 
                    next_Q3 = 0
                
                if Q4A[-1] + gQ3*P - gQ4 + (rgQ3*P)*gQ4 > 0:
                    next_Q4A = Q4A[-1] + gQ3*P - gQ4 + (rgQ3*P)*gQ4
                    
                else:
                    next_Q4A = 0
        
                if Q2A[-1] + gQ4 + gQ3*(1-P) - gQ2 + (rgQ3*(1-P) - rgQ2)*gQ4 > 0:
                    next_Q2A = Q2A[-1] + gQ4 + gQ3*(1-P) - gQ2 + (rgQ3*(1-P) - rgQ2)*gQ4
                    
                else:
                    next_Q2A = 0
                
                if Q1AA[-1] + gQ2 - gQ1 + (rgQ2 - rgQ1)*gQ4 < 0:
                    next_Q1AA= 0
        
                else: 
                    next_Q1AA= Q1AA[-1] + gQ2 - gQ1 + (rgQ2 - rgQ1)*gQ4
                    
        
                if Q0AAA[-1] + gQ1 + (rgQ1)*gQ4 < 0:
                    next_Q0AAA = 0
                    
                else: 
                    next_Q0AAA = Q0AAA[-1] + gQ1 + (rgQ1)*gQ4
                
                if Al5[-1] - gAl5 + (-rgAl5)*gQ4 > 0:
                    next_Al5 = Al5[-1] - gAl5 + (-rgAl5)*gQ4
                else:
                    next_Al5 = 0
                
                if Al4A[-1] + gAl5 + (rgAl5)*gQ4 > 0:
                    next_Al4A = Al4A[-1] + gAl5 + (rgAl5)*gQ4
                else:
                    next_Al4A = 0
                    
                next_Al_draw = Al_draw[-1]+1
                Al_draw.append(next_Al_draw)
                
                Q3.append(next_Q3)
                Q4A.append(next_Q4A)
                Q2A.append(next_Q2A)  
                Q1AA.append(next_Q1AA)
                Q0AAA.append(next_Q0AAA)
        
                Al5.append(next_Al5)
                Al4A.append(next_Al4A)
                
            ## Residual Al draw ##
            aa = ((-300*(Al5[-1]*100/(100-Al4A[-1])))/(-100+(Al5[-1]*100/(100-Al4A[-1]))))-(-300*(Al5[-2]*100/(100-Al4A[-2])))/(-100+(Al5[-2]*100/(100-Al4A[-2])))
            b = -aa*Al_draw[-1]+(-300*(Al5[-1]*100/(100-Al4A[-1])))/(-100+(Al5[-1]*100/(100-Al4A[-1])))
            balance_draw = (-b/(aa-1))-Al_draw[-2]
               
            gQ3 = balance_draw*(Q3[-2]*w[6])*(1/(R_data[R_ind]+1)) / (Q3[-2]*w[6]+Q4A[-2]*w[7]+Q2A[-2]*w[8]+Q1AA[-2]*w[9]+Al5[-2]*w[10])
            gQ4 = balance_draw*(Q4A[-2]*w[7])*(1/(R_data[R_ind]+1)) / (Q3[-2]*w[6]+Q4A[-2]*w[7]+Q2A[-2]*w[8]+Q1AA[-2]*w[9]+Al5[-2]*w[10])
            gQ2 = balance_draw*(Q2A[-2]*w[8])*(1/(R_data[R_ind]+1)) / (Q3[-2]*w[6]+Q4A[-2]*w[7]+Q2A[-2]*w[8]+Q1AA[-2]*w[9]+Al5[-2]*w[10])
            gQ1 = balance_draw*(Q1AA[-2]*w[9])*(1/(R_data[R_ind]+1)) / (Q3[-2]*w[6]+Q4A[-2]*w[7]+Q2A[-2]*w[8]+Q1AA[-2]*w[9]+Al5[-2]*w[10])
            gAl5 = balance_draw*(Al5[-2]*w[10])*(1/(R_data[R_ind]+1)) / (Q3[-2]*w[6]+Q4A[-2]*w[7]+Q2A[-2]*w[8]+Q1AA[-2]*w[9]+Al5[-2]*w[10])
            
            rgQ3 = gQ3 / (gQ3 + gQ2 + gQ1 + gAl5)
            rgQ2 = gQ2 / (gQ3 + gQ2 + gQ1 + gAl5)
            rgQ1 = gQ1 / (gQ3 + gQ2 + gQ1 + gAl5)
            rgAl5 = gAl5 / (gQ3 + gQ2 + gQ1 + gAl5)
            
            #overgang
            if (Q4A[-2] + Q2A[-2] + 2*Q1AA[-2] + 3*Q0AAA[-2]) < Q3[0]*w[5]:
                P = 1
            else:
                P = 0
            
            # Draws        
            if Q3[-2] - gQ3 + (- rgQ3)*gQ4 > 0:
                next_Q3 = Q3[-2] - gQ3 + (- rgQ3)*gQ4
                
            else: 
                next_Q3 = 0
            
            if Q4A[-2] + gQ3*P - gQ4 + (rgQ3*P)*gQ4 > 0:
                next_Q4A = Q4A[-2] + gQ3*P - gQ4 + (rgQ3*P)*gQ4
                
            else:
                next_Q4A = 0
        
            if Q2A[-2] + gQ4 + gQ3*(1-P) - gQ2 + (rgQ3*(1-P) - rgQ2)*gQ4 > 0:
                next_Q2A = Q2A[-2] + gQ4 + gQ3*(1-P) - gQ2 + (rgQ3*(1-P) - rgQ2)*gQ4
                
            else:
                next_Q2A = 0
            
            if Q1AA[-2] + gQ2 - gQ1 + (rgQ2 - rgQ1)*gQ4 < 0:
                next_Q1AA= 0
        
            else: 
                next_Q1AA= Q1AA[-2] + gQ2 - gQ1 + (rgQ2 - rgQ1)*gQ4
                
        
            if Q0AAA[-2] + gQ1 + (rgQ1)*gQ4 < 0:
                next_Q0AAA = 0
                
            else: 
                next_Q0AAA = Q0AAA[-2] + gQ1 + (rgQ1)*gQ4
            
            if Al5[-2] - gAl5 + (-rgAl5)*gQ4 > 0:
                next_Al5 = Al5[-2] - gAl5 + (-rgAl5)*gQ4
            else:
                next_Al5 = 0
            
            if Al4A[-2] + gAl5 + (rgAl5)*gQ4 > 0:
                next_Al4A = Al4A[-2] + gAl5 + (rgAl5)*gQ4
            else:
                next_Al4A = 0
            
            Q3.append(next_Q3)
            Q4A.append(next_Q4A)
            Q2A.append(next_Q2A)
            Q1AA.append(next_Q1AA)
            Q0AAA.append(next_Q0AAA)
            Al5.append(next_Al5)
            Al4A.append(next_Al4A)
            
            Q3Na.append(next_Q3)
            Al5Na.append(next_Al5)
            Al4Na.append(next_Al4A)
            Q4Na.append(next_Q4A)
            Q2Na.append(next_Q2A)
            Q1Na.append(next_Q1AA)
            Q0Na.append(next_Q0AAA)
                
        #For loop for udregning af g og Q_Al5
        for i in list(range(int(-(100*mod_data[R_ind])/(mod_data[R_ind]-100))+1)):
        
            if Q4A[-1] + Q4N[-1] > 0:
                Q4 = Q4A[-1] + Q4N[-1]
            else:
                Q4 = 0
                
            if Q2A[-1] + Q2N[-1] > 0:
                Q2 = Q2A[-1] + Q2N[-1]
            else:
                Q2 = 0
            
            if Q1AA[-1] + Q1AN[-1] + Q1NN[-1] > 0:
                Q1 = Q1AA[-1] + Q1AN[-1] + Q1NN[-1]
            else:
                Q1 = 0
                
            if Q0AAA[-1] + Q0AAN[-1] + Q0ANN[-1] + Q0NNN[-1] > 0:
                Q0 = Q0AAA[-1] + Q0AAN[-1] + Q0ANN[-1] + Q0NNN[-1]
            else:
                Q0 = 0
                
            if Al4A[-1] + Al4N[-1] > 0:
                Al4 = Al4A[-1] + Al4N[-1]
            else:
                Al4 = 0
                
         
            ## M2O DRAW ##
            ipQ4 = (Q4A[-1]*iw[6])                                   / ((Q4A[-1]*iw[6]) + (Q2A[-1]*iw[6]) + ((Q1AA[-1]*2 + Q1AN[-1])*iw[8]) + ((Q0AAA[-1]*3 + Q0AAN[-1]*2 + Q0ANN[-1])*iw[9]) + (Al4A[-1]*w[11]))
            ipQ2 = (Q2A[-1]*iw[6])                                   / ((Q4A[-1]*iw[6]) + (Q2A[-1]*iw[6]) + ((Q1AA[-1]*2 + Q1AN[-1])*iw[8]) + ((Q0AAA[-1]*3 + Q0AAN[-1]*2 + Q0ANN[-1])*iw[9]) + (Al4A[-1]*w[11]))
            ipQ1 = ((Q1AA[-1]*2 + Q1AN[-1])*iw[8])                   / ((Q4A[-1]*iw[6]) + (Q2A[-1]*iw[6]) + ((Q1AA[-1]*2 + Q1AN[-1])*iw[8]) + ((Q0AAA[-1]*3 + Q0AAN[-1]*2 + Q0ANN[-1])*iw[9]) + (Al4A[-1]*w[11]))
            ipQ0 = ((Q0AAA[-1]*3 + Q0AAN[-1]*2 + Q0ANN[-1])*iw[9])   / ((Q4A[-1]*iw[6]) + (Q2A[-1]*iw[6]) + ((Q1AA[-1]*2 + Q1AN[-1])*iw[8]) + ((Q0AAA[-1]*3 + Q0AAN[-1]*2 + Q0ANN[-1])*iw[9]) + (Al4A[-1]*w[11]))
            ipAl4 = (Al4A[-1]*w[11])                                 / ((Q4A[-1]*iw[6]) + (Q2A[-1]*iw[6]) + ((Q1AA[-1]*2 + Q1AN[-1])*iw[8]) + ((Q0AAA[-1]*3 + Q0AAN[-1]*2 + Q0ANN[-1])*iw[9]) + (Al4A[-1]*w[11]))


            Alcoef = 1/(1 + 3*ipAl4)   
            
            pQ3 = (Q3[-1]*w[0]) / (Q3[-1]*w[0] + Q4*w[1] + Q2*w[2] + Q1*w[3] + Al5[-1]*w[4])
            pQ4 = (Q4*w[1]) / (Q3[-1]*w[0] + Q4*w[1] + Q2*w[2] + Q1*w[3] + Al5[-1]*w[4])
            pQ2 = (Q2*w[2]) / (Q3[-1]*w[0] + Q4*w[1] + Q2*w[2] + Q1*w[3] + Al5[-1]*w[4])
            pQ1 = (Q1*w[3]) / (Q3[-1]*w[0] + Q4*w[1] + Q2*w[2] + Q1*w[3] + Al5[-1]*w[4])
            pAl5 = Alcoef*(Al5[-1]*w[4]) / (Q3[-1]*w[0] + Q4*w[1] + Q2*w[2] + Q1*w[3] + Al5[-1]*w[4])


            
            rpQ3 = pQ3 / (pQ3 + pQ2 + pQ1 + pAl5)
            rpQ2 = pQ2 / (pQ3 + pQ2 + pQ1 + pAl5)
            rpQ1 = pQ1 / (pQ3 + pQ2 + pQ1 + pAl5)
            rpAl5 = pAl5 / (pQ3 + pQ2 + pQ1 + pAl5)
                
            if Q4 > 0:
                rQ4A = Q4A[-1]/(Q4A[-1]+Q4N[-1])
                rQ4N = Q4N[-1]/(Q4A[-1]+Q4N[-1])
            else:
                rQ4A = 0
                rQ4N = 0
                
            if Q2 > 0:
                rQ2A = Q2A[-1]/(Q2A[-1]+Q2N[-1])
                rQ2N = Q2N[-1]/(Q2A[-1]+Q2N[-1])
            else:
                rQ2A = 0
                rQ2N = 0
            
            if Q1 > 0:
                rQ1AA = Q1AA[-1]/(Q1AA[-1]+Q1AN[-1]+Q1NN[-1])
                rQ1AN = Q1AN[-1]/(Q1AA[-1]+Q1AN[-1]+Q1NN[-1])
                rQ1NN = Q1NN[-1]/(Q1AA[-1]+Q1AN[-1]+Q1NN[-1])
            else:
                rQ1AA = 0
                rQ1AN = 0
                rQ1NN = 0
                    
        
            if Q1AA[-1]+Q1AN[-1] > 0:
                irQ1AA = Q1AA[-1]*2/(Q1AA[-1]*2+Q1AN[-1])
                irQ1AN = Q1AN[-1]/(Q1AA[-1]*2+Q1AN[-1])
                
            else:
                irQ1AA = 0
                irQ1AN = 0
        
            if Q0AAA[-1] + Q0AAN[-1] + Q0ANN[-1] > 0:
                irQ0AAA = Q0AAA[-1]*3/(Q0AAA[-1]*3 + Q0AAN[-1]*2 + Q0ANN[-1])
                irQ0AAN = Q0AAN[-1]*2/(Q0AAA[-1]*3 + Q0AAN[-1]*2 + Q0ANN[-1])
                irQ0ANN = Q0ANN[-1]/(Q0AAA[-1]*3 + Q0AAN[-1]*2 + Q0ANN[-1])
                
            else:
                irQ0AAA = 0
                irQ0AAN = 0
                irQ0ANN = 0
                
           
            # Draws
            if Q4 + Q2 + 2*Q1 + 3*Q0 < Q3[0]*w[5]:
                P = 1
            else:
                P = 0
                
            if Q3[-1] - pQ3 + 3*pAl5*(ipQ4 + ipQ2) + (- rpQ3 + 3*rpAl5*(ipQ4 + ipQ2))*pQ4 > 0:
                next_Q3 = Q3[-1] - pQ3 + 3*pAl5*(ipQ4 + ipQ2) + (- rpQ3 + 3*rpAl5*(ipQ4 + ipQ2))*pQ4
            else:
                next_Q3 = 0
   
                     
            if Q4N[-1] + pQ3*P - pQ4*rQ4N + (rpQ3*P)*pQ4 > 0:
                next_Q4N = Q4N[-1] + pQ3*P - pQ4*rQ4N + (rpQ3*P)*pQ4
            else:
                next_Q4N = 0
                
            if Q4A[-1] - pQ4*rQ4A - 3*pAl5*ipQ4 + (- 3*rpAl5*ipQ4)*pQ4 > 0:
                next_Q4A = Q4A[-1] - pQ4*rQ4A - 3*pAl5*ipQ4 + (- 3*rpAl5*ipQ4)*pQ4
            else:
                next_Q4A = 0
       
         
            if Q2A[-1] + pQ4*rQ4A - pQ2*rQ2A - 3*pAl5*ipQ2 + 3*pAl5*ipQ1*irQ1AA + (-rpQ2*rQ2A - 3*rpAl5*ipQ2 + 3*rpAl5*ipQ1*irQ1AA)*pQ4 > 0:
                next_Q2A = Q2A[-1] + pQ4*rQ4A - pQ2*rQ2A - 3*pAl5*ipQ2 + 3*pAl5*ipQ1*irQ1AA + (- rpQ2*rQ2A - 3*rpAl5*ipQ2 + 3*rpAl5*ipQ1*irQ1AA)*pQ4
            else:
                next_Q2A = 0

            if Q2N[-1] + pQ4*rQ4N - pQ2*rQ2N + pQ3*(1-P) + 3*pAl5*ipQ1*irQ1AN + (-rpQ2*rQ2N + rpQ3*(1-P) + 3*rpAl5*ipQ1*irQ1AN)*pQ4 > 0:
                next_Q2N = Q2N[-1] + pQ4*rQ4N - pQ2*rQ2N + pQ3*(1-P) + 3*pAl5*ipQ1*irQ1AN + (- rpQ2*rQ2N + rpQ3*(1-P) + 3*rpAl5*ipQ1*irQ1AN)*pQ4                         
            else:
                next_Q2N = 0
                    
                
            if Q1AA[-1] - pQ1*rQ1AA + 3*pAl5*ipQ0*irQ0AAA - 3*pAl5*ipQ1*irQ1AA + (- rpQ1*rQ1AA + 3*rpAl5*ipQ0*irQ0AAA - 3*rpAl5*ipQ1*irQ1AA)*pQ4 < 0:
                next_Q1AA = 0
            else: 
                next_Q1AA = Q1AA[-1] - pQ1*rQ1AA + 3*pAl5*ipQ0*irQ0AAA - 3*pAl5*ipQ1*irQ1AA + (- rpQ1*rQ1AA + 3*rpAl5*ipQ0*irQ0AAA - 3*rpAl5*ipQ1*irQ1AA)*pQ4

            if Q1AN[-1] + pQ2*rQ2A - pQ1*rQ1AN + 3*pAl5*ipQ0*irQ0AAN - 3*pAl5*ipQ1*irQ1AN + (rpQ2*rQ2A - rpQ1*rQ1AN + 3*rpAl5*ipQ0*irQ0AAN - 3*rpAl5*ipQ1*irQ1AN)*pQ4 < 0:
                next_Q1AN = 0                
            else: 
                next_Q1AN = Q1AN[-1] + pQ2*rQ2A - pQ1*rQ1AN + 3*pAl5*ipQ0*irQ0AAN - 3*pAl5*ipQ1*irQ1AN + (rpQ2*rQ2A - rpQ1*rQ1AN + 3*rpAl5*ipQ0*irQ0AAN - 3*rpAl5*ipQ1*irQ1AN)*pQ4
        
            if Q1NN[-1] + pQ2*rQ2N - pQ1*rQ1NN + 3*pAl5*ipQ0*irQ0ANN + (rpQ2*rQ2N - rpQ1*rQ1NN + 3*rpAl5*ipQ0*irQ0ANN)*pQ4 < 0:
                next_Q1NN = 0
            else: 
                next_Q1NN = Q1NN[-1] + pQ2*rQ2N - pQ1*rQ1NN + 3*pAl5*ipQ0*irQ0ANN + (rpQ2*rQ2N - rpQ1*rQ1NN + 3*rpAl5*ipQ0*irQ0ANN)*pQ4


            if Q0AAA[-1] - 3*pAl5*ipQ0*irQ0AAA + (- 3*rpAl5*ipQ0*irQ0AAA)*pQ4 < 0:
                next_Q0AAA = 0         
            else:
                next_Q0AAA = Q0AAA[-1] - 3*pAl5*ipQ0*irQ0AAA + (-3*rpAl5*ipQ0*irQ0AAA)*pQ4

            if Q0AAN[-1] + pQ1*rQ1AA - 3*pAl5*ipQ0*irQ0AAN + (rpQ1*rQ1AA - 3*rpAl5*ipQ0*irQ0AAN)*pQ4 < 0:
                next_Q0AAN = 0        
            else:
                next_Q0AAN = Q0AAN[-1] + pQ1*rQ1AA - 3*pAl5*ipQ0*irQ0AAN + (rpQ1*rQ1AA - 3*rpAl5*ipQ0*irQ0AAN)*pQ4

            if Q0ANN[-1] + pQ1*rQ1AN - 3*pAl5*ipQ0*irQ0ANN + (rpQ1*rQ1AN - 3*rpAl5*ipQ0*irQ0ANN)*pQ4 < 0:
                next_Q0ANN = 0       
            else:
                next_Q0ANN = Q0ANN[-1] + pQ1*rQ1AN - 3*pAl5*ipQ0*irQ0ANN + (rpQ1*rQ1AN - 3*rpAl5*ipQ0*irQ0ANN)*pQ4

            if Q0NNN[-1] + pQ1*rQ1NN + (rpQ1*rQ1NN)*pQ4 < 0:
                next_Q0NNN = 0
            else:
                next_Q0NNN = Q0NNN[-1] + pQ1*rQ1NN + (rpQ1*rQ1NN)*pQ4
            
            if Al5[-1] - pAl5 - pQ4*rpAl5 > 0:
                next_Al5 = Al5[-1] - pAl5 - pQ4*rpAl5
            else: 
                next_Al5 = 0
            
            if Al4A[-1] - 3*pAl5*ipAl4 + (- 3*rpAl5*ipAl4)*pQ4 > 0:
                next_Al4A = Al4A[-1] - 3*pAl5*ipAl4 + (- 3*rpAl5*ipAl4)*pQ4
            else:
                next_Al4A = 0

            if Al4N[-1] + pAl5 + 3*pAl5*ipAl4 + (rpAl5 + 3*rpAl5*ipAl4)*pQ4 > 0:
                next_Al4N = Al4N[-1] + pAl5 + 3*pAl5*ipAl4 + (rpAl5 + 3*rpAl5*ipAl4)*pQ4
            else:
                next_Al4N = 0

            
            Q3shinanigens.append(- 3*pAl5*ipAl4 + (- 3*rpAl5*ipAl4)*pQ4 + 3*pAl5*ipAl4 + (3*rpAl5*ipAl4)*pQ4)   
            
            
            Q3.append(next_Q3)
    
            Q4A.append(next_Q4A)
            Q4N.append(next_Q4N)
    
            Q2A.append(next_Q2A)
            Q2N.append(next_Q2N)
    
            Q1AA.append(next_Q1AA)
            Q1AN.append(next_Q1AN)
            Q1NN.append(next_Q1NN)              

            Q0AAA.append(next_Q0AAA)
            Q0AAN.append(next_Q0AAN)
            Q0ANN.append(next_Q0ANN)
            Q0NNN.append(next_Q0NNN)
    
            Al5.append(next_Al5)
            
            Al4A.append(next_Al4A)
            Al4N.append(next_Al4N)
            
            Q3Na.append(next_Q3)
            Al5Na.append(next_Al5)
            Al4Na.append(next_Al4A + next_Al4N)
            Q4Na.append(next_Q4A + next_Q4N)
            Q2Na.append(next_Q2A + next_Q2N)
            Q1Na.append(next_Q1AA + next_Q1AN + next_Q1NN)
            Q0Na.append(next_Q0AAA + next_Q0AAN + next_Q0ANN + next_Q0NNN)
        
            
        
        # Vi laver lister over teoretiske værdier udregnet fra modellen
        mod_m = min(M2O, key=lambda x:abs(x-mod_data[R_ind]))
        ind = M2O.index(mod_m)
        Q3_m.append(Q3Na[ind])
        Q4_m.append(Q4Na[ind])
        Q2_m.append(Q2Na[ind])
        Q1_m.append(Q1Na[ind])
        Q0_m.append(Q0Na[ind])
        Al5_m.append(Al5Na[ind])
        Al4_m.append(Al4Na[ind])

    Q3_m = np.array(Q3_m)
    Q4_m = np.array(Q4_m)
    Q2_m = np.array(Q2_m)
    Q1_m = np.array(Q1_m)
    Q0_m = np.array(Q0_m)
    Al5_m = np.array(Al5_m)
    Al4_m = np.array(Al4_m)
    
    if SSEtrue == 1:
        SSE = sum(((Q4_data - Q4_m)**2) + ((Q3_data - Q3_m)**2) + ((Q2_data - Q2_m)**2) + ((Q1_data - Q1_m)**2) + ((Q0_data - Q0_m)**2) + ((Al4_data - Al4_m)**2) + ((Al5_data - Al5_m)**2))
        return SSE
    else:
        return 0



SSElist =[]
it = 1
def print_SSE(x, f, accepted):
    global it
    global SSElist 
    print("SSE = {} for {}".format(f, it))
    print(x)
    it += 1
    SSElist.append(f)

w1 = [0.563127597849139, 0.00450496165645882, 13.56029962451884, 22.781360649194546, 0.37113618230937606, 4.516182612518033, 2.329984806723401, 0.4191801056312357, 0.6126909009591526, 0.9541238974724411, 0.5341439674805613]

#res = scipy.optimize.basinhopping(model, w1, niter=500, T=1.0, stepsize=0.5, minimizer_kwargs=None, take_step=None, accept_test=None, callback=print_SSE, interval=50, disp=False, niter_success=None)
#
#w = [abs(res.x[0]), abs(res.x[1]), abs(res.x[2]), abs(res.x[3]), abs(res.x[4]),  abs(res.x[5]), abs(res.x[6]),  abs(res.x[7]),  abs(res.x[8]),  abs(res.x[9]), abs(res.x[10])]

#SSE = 338 for 500 iter
#w = [0.8931466395555216, 0.006664939342189158, 0.011811319068313344, 14.717899967706934, 0.3252592331110711, 3.8647635051715974, 2.23492900617967, 0.4824790404940299, 1.187200882367413, 1.0696818238399677, 4.4908085391780865]

w = [0.8931466395555216, 0.006664939342189158, 0.011811319068313344, 14.717899967706934, 0.3252592331110711, 3.8647635051715974, 2.23492900617967, 0.4824790404940299, 1.187200882367413, 1.0696818238399677, 4.4908085391780865]
model(w)

for i9 in range(100):
    Al5Na.append(0)
    Q1AN.append(0)
    Q0ANN.append(0)
    Q0AAN.append(0)
    Al4Na.append(0)
    Q4Na.append(0)
    Q2Na.append(0)
    Q3Na.append(0)
    Q1Na.append(0)
    Q0Na.append(0)

fig1 = plt.figure(figsize=(12, 7))
plt.subplot(111)
plt.plot(M2O[0:100], Q4A[-101:-1], 'y', M2O[0:100], Q2A[-101:-1], 'g', 
         M2O[0:100], Al4A[-101:-1], 'm', M2O[0:100],  Al5Na[0:100], 'b', M2O[0:100], 
         Q1AN[0:100], 'c', M2O[0:100], Q1AA[-101:-1], 'c', M2O[0:100], Q0ANN[0:100], 'k', 
         M2O[0:100], Q0AAN[0:100], 'k', M2O[0:100], Q0AAA[-101:-1], 'k')
plt.xlabel("M2O (mol %)")
plt.ylabel("Qn/Al distribution (mol %)")
plt.title('Distribution')
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
            wspace=0.35)
        

mod_data = np.array(mod_data)   
Al5_m = np.array(Al5_m)   
Al4_m = np.array(Al4_m)   
Q4_m = np.array(Q4_m)   
Q2_m = np.array(Q2_m)   
Q1_m = np.array(Q1_m)   
Q0_m = np.array(Q0_m)   
 
M2O = np.array(M2O) 
Al5Na = np.array(Al5Na)   
Al4Na = np.array(Al4Na)   
Q4Na = np.array(Q4Na)   
Q2Na = np.array(Q2Na)   
Q1Na = np.array(Q1Na)   
Q0Na = np.array(Q0Na) 


Q4A = np.array(Q4A)
Q2A = np.array(Q2A)
Q1AA =np.array(Q1AA)
Q0AAA = np.array(Q0AAA) 
    
Al5 = np.array(Al5)
Al4A = np.array(Al4A)

ladningGlas = (mod_data/(100 - mod_data))*100 + 3*Al5_m - Al4_m - Q4_m - Q2_m - 2*Q1_m - 3*Q0_m
#ladningNa = (M2O[0:100]/(100 - M2O[0:100]))*100 + 3*Al5Na - Al4Na - Q4Na - Q2Na - 2*Q1Na - 3*Q0Na
ladningAl = (3*Al5[0:len(Al_draw)+1] - Al4A[0:len(Al_draw)+1] - Q4A[0:len(Al_draw)+1]- Q2A[0:len(Al_draw)+1] - 2*Q1AA[0:len(Al_draw)+1] - 3*Q0AAA[0:len(Al_draw)+1])
#ladningAl = (- Al4A[0:len(Al_draw)+1] - Q4A[0:len(Al_draw)+1]- Q2A[0:len(Al_draw)+1] - 2*Q1AA[0:len(Al_draw)+1] - 3*Q0AAA[0:len(Al_draw)+1])
MasseAl = (Al4A[0:len(Al_draw)+1] + Al5[0:len(Al_draw)+1] + Q3[0:len(Al_draw)+1] + Q4A[0:len(Al_draw)+1] + Q2A[0:len(Al_draw)+1] + Q1AA[0:len(Al_draw)+1] + Q0AAA[0:len(Al_draw)+1])
MasseNa = (Al5Na + Al4Na + Q4Na + Q3Na + Q2Na + Q1Na + Q0Na)

ladningNa = (M2O[0:100]/(100 - M2O[0:100]))*100 + 3*Al5Na[0:100] - Al4Na[0:100] - Q4Na[0:100] - Q2Na[0:100] - 2*Q1Na[0:100] - 3*Q0Na[0:100]
ladningNegativ = (- Al4Na - Q4Na - Q2Na - 2*Q1Na - 3*Q0Na)


#RMSElist = []
#
#for i in SSElist:
#    RMSE = math.sqrt(i/(7*len(R_data)))
#    RMSElist.append(RMSE)
#
#tab = tt.Texttable()
#headings = ['W','Al5','Na']
#tab.header(headings)
#names = ['Q3', 'Q4', 'Q2', 'Q1', 'Al5/6' ]
#Al5 = [w[6], w[7], w[8], w[9], w[10]]
#Na = [w[0], w[1], w[2], w[3], w[4]]
#for row in zip(names,Al5,Na):
#    tab.add_row(row)
#s = tab.draw()
#print (s)



#with open('Results_{}_{}_{}.csv'.format(file_name, fil, tid), 'w', newline = '') as csvfile:
#    filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
#    filewriter.writerow(w)
#    for a in list(range(len(SSElist))):
#        filewriter.writerow((SSElist[a], RMSElist[a]))
        
## Herfra plotter vi ##

# Startværdier


#with open('Model_data_{}_{}_{}.csv'.format(file_name, fil, tid), 'w', newline = '') as csvfile:
#    filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
#    for a in list(range(len(Q3_m))):
#        filewriter.writerow((Q3_m[a], Q4_m[a], Q2_m[a], Q1_m[a], Q0_m[a], Al5_m[a], Al4_m[a]))
        
        
        
a = [0, 80]
b = [0, 80]
fitQ3 = np.polyfit(a, b, 1)
fit_fn = np.poly1d(fitQ3) 

font = FontProperties()
font.set_family('serif')
font.set_name('Palatino Linotype')
font.set_size('14')

matplotlib.rcParams.update(
    {
        'text.usetex': False,
        'font.family': 'Palatino Linotype ',
        'mathtext.fontset': 'cm',
        'font.size': 14
    }
)


#if fil == "Kapoor_2018":
#    
#    pltAl = [Al_data[0], Al_data[1], Al_data[2], Al_data[3], Al_data[4], Al_data[5]]
#    pltNa = [mod_data[6], mod_data[7], mod_data[8], mod_data[9], mod_data[10], mod_data[11], mod_data[12]]
#    
#    Al203 = np.array(pltAl)
#    Na2O = np.array(pltNa)
#    
#    fig = plt.figure(figsize=(14, 7))
#    
#    plt.subplot(131)
#    b3, b4, b2, b1, b0, a4, a5 = plt.plot(Q3_data, Q3_m,'ro', Q4_data, Q4_m , 'ys', Q2_data, Q2_m, 'gd', Q1_data , Q1_m, 'cp', Q0_data, Q0_m, 'k*', Al4_data, Al4_m , 'm^', Al5_data, Al5_m , 'bv')
#    plt.plot(a ,fit_fn(a),':k')
#    plt.xlabel("Experimental (%)", fontproperties=font)
#    plt.ylabel("Model (%)", fontproperties=font)
#    #plt.legend(handles=[Q3_red, Q4_yellow, Q2_green, Al5_blue, Al4_magenta, Q1_cyan, Q0_black])
#    
#    plt.subplot(132)
#    plt.plot(Al203, Q3_m[0:6], 'r', Al203, Q4_m[0:6], 'y', Al203, Q2_m[0:6], 'g', Al203, Al4_m[0:6], 'm', Al203, Al5_m[0:6], 'b', Al203, Q1_m[0:6], 'c', Al203, Q0_m[0:6], 'k',
#             Al203, Q3_data[0:6], 'ro', Al203, Q4_data[0:6], 'ys', Al203, Q2_data[0:6], 'gd', Al203, Al4_data[0:6], 'm^', Al203, Al5_data[0:6], 'bv')
#    plt.xlabel("Al$_2$O$_3$ (mol%)", fontproperties=font)
#    plt.ylabel("Species fraction (%)", fontproperties=font)
#    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
#                        wspace=0.35)
#    
#    plt.subplot(133)
#    plt.plot(Na2O, Q3_m[6:13], 'r', Na2O, Q4_m[6:13], 'y', Na2O, Q2_m[6:13], 'g', Na2O, Al4_m[6:13], 'm', Na2O, Al5_m[6:13], 'b', Na2O, Q1_m[6:13], 'c', Na2O, Q0_m[6:13], 'k',
#             Na2O, Q3_data[6:13], 'ro', Na2O, Q4_data[6:13], 'ys', Na2O, Q2_data[6:13], 'gd', Na2O, Al4_data[6:13], 'm^', Na2O, Al5_data[6:13], 'bv')
#    plt.xlabel("Na$_2$O (mol%)", fontproperties=font)
#    plt.ylabel("Species fraction (%)", fontproperties=font)
#    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.91, hspace=0.25,
#                        wspace=0.35)
#    fig.legend((b3, b4, b2, b1, b0, a5, a4), ('Q$^3$', 'Q$^4$', 'Q$^2$', 'Q$^1$','Q$^0$', 'Al$^5$', 'Al$^4$'), loc = (0.92, 0.4))
#    plt.savefig("{}_{}_{}.svg".format(file_name, fil, tid))

#else:
#fig = plt.figure(figsize=(7.5, 7))
#
#b3, b4, b2, b1, b0, a4, a5 = plt.plot(Q3_data, Q3_m, 'ro', Q4_data, Q4_m , 'ys', Q2_data, Q2_m, 'gd', Q1_data , Q1_m, 'cp', Q0_data, Q0_m, 'k*', Al4_data, Al4_m , 'm^', Al5_data, Al5_m , 'bv' )
#plt.plot(a ,fit_fn(a),':k')
#plt.xlabel("Experimental (%)", fontproperties=font)
#plt.ylabel("Model (%)", fontproperties=font)
#plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.85, hspace=0.25,
#                    wspace=0.35)
#fig.legend((b3, b4, b2, b1, b0, a5, a4), ('Q$^3$', 'Q$^4$', 'Q$^2$', 'Q$^1$','Q$^0$', 'Al$^5$', 'Al$^4$'), loc = (0.86, 0.4))
#    plt.savefig("{}_{}_{}.svg".format(file_name, fil, tid))


Al_series_Al = list(range(5,27))
Al_series_Al = np.array(Al_series_Al)
Al_series_B = 80-Al_series_Al
Al_series_Na = [20]*22
Al_series_B = np.array(Al_series_B)
Al_series_Na = np.array(Al_series_Na)

Na_series_Na = np.arange(15,50.5,0.5)
Na_series_B = 80-Na_series_Na
Na_series_Al = [20]*71
Na_series_B = np.array(Na_series_B)
Na_series_Al = np.array(Na_series_Al)

Al_series_R = Al_series_Al / Al_series_B
Na_series_R = Na_series_Al / Na_series_B

Na = np.concatenate([Al_series_Na, Na_series_Na])


mod_data = np.array(mod_data)
R_data = np.array(R_data)


mod_data = np.concatenate([mod_data, Al_series_Na, Na_series_Na]).tolist()
R_data = np.concatenate([R_data, Al_series_R, Na_series_R]).tolist()
M2O = np.array(M2O).tolist()

SSEtrue = 2


model(w) 

fig1 = plt.figure(figsize=(12, 7))
plt.subplot(111)
plt.plot(M2O, Q3Na, 'r', M2O, Q4Na, 'y', M2O, Q2Na, 'g', M2O, Al4Na, 'm', M2O,  Al5Na, 'b', M2O, Q1Na, 'c', M2O, Q0Na, 'k')
plt.xlabel("M2O (mol %)")
plt.ylabel("Qn/Al distribution (mol %)")
plt.title('Distribution')
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
            wspace=0.35)


if fil == "Kapoor_2018":
    
    pltAl = [Al_data[0], Al_data[1], Al_data[2], Al_data[3], Al_data[4]]
    pltNa = [mod_data[5], mod_data[6], mod_data[7], mod_data[8], mod_data[9], mod_data[10], mod_data[11]]
    
    Al203 = np.array(pltAl)
    Na2O = np.array(pltNa)
    
    fig = plt.figure(figsize=(14, 7))
    
    plt.subplot(131)
    b3, b4, b2, b1, b0, a4, a5 = plt.plot(Q3_data, Q3_m[0:12],'ro', Q4_data, Q4_m[0:12] , 'ys', Q2_data, Q2_m[0:12], 'gd', Q1_data , Q1_m[0:12], 'cp', Q0_data, Q0_m[0:12], 'k*', Al4_data, Al4_m[0:12] , 'm^', Al5_data, Al5_m[0:12] , 'bv')
    plt.plot(a ,fit_fn(a),':k')
    plt.xlabel("Experimental (%)", fontproperties=font)
    plt.ylabel("Model (%)", fontproperties=font)
    #plt.legend(handles=[Q3_red, Q4_yellow, Q2_green, Al5_blue, Al4_magenta, Q1_cyan, Q0_black])
    
    plt.subplot(132)
    plt.plot(Al_series_Al, Q3_m[12:34], 'r', Al_series_Al, Q4_m[12:34], 'y', Al_series_Al, Q2_m[12:34], 'g', Al_series_Al, Al4_m[12:34], 'm', Al_series_Al, Al5_m[12:34], 'b', Al_series_Al, Q1_m[12:34], 'c', Al_series_Al, Q0_m[12:34], 'k',
             Al203, Q3_data[0:5], 'ro', Al203, Q4_data[0:5], 'ys', Al203, Q2_data[0:5], 'gd', Al203, Al4_data[0:5], 'm^', Al203, Al5_data[0:5], 'bv')
    plt.xlabel("Al$_2$O$_3$ (mol%)", fontproperties=font)
    plt.ylabel("Species fraction (%)", fontproperties=font)
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                        wspace=0.35)
    
    plt.subplot(133)
    plt.plot(Na_series_Na, Q3_m[34:105], 'r', Na_series_Na, Q4_m[34:105], 'y', Na_series_Na, Q2_m[34:105], 'g', Na_series_Na, Al4_m[34:105], 'm', Na_series_Na, Al5_m[34:105], 'b', Na_series_Na, Q1_m[34:105], 'c', Na_series_Na, Q0_m[34:105], 'k',
             Na2O, Q3_data[5:12], 'ro', Na2O, Q4_data[5:12], 'ys', Na2O, Q2_data[5:12], 'gd', Na2O, Al4_data[5:12], 'm^', Na2O, Al5_data[5:12], 'bv')
    plt.xlabel("Na$_2$O (mol%)", fontproperties=font)
    plt.ylabel("Species fraction (%)", fontproperties=font)
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.91, hspace=0.25,
                        wspace=0.35)
    fig.legend((b3, b4, b2, b1, b0, a5, a4), ('Q$^3$', 'Q$^4$', 'Q$^2$', 'Q$^1$','Q$^0$', 'Al$^5$', 'Al$^4$'), loc = (0.92, 0.4))