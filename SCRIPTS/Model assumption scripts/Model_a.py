import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import csv
import scipy.optimize
import math
import os
import texttable as tt
import datetime
import sys

file_name =  os.path.basename(sys.argv[0])
file_name = file_name[:-3]
tid = datetime.datetime.now().strftime("%d-%m-%y_%H-%M") 

# Vi laver lister ud fra det eksperimentielle data fra vores csv fil
fil="Samlet_Na"
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
mod_data = [float(i) for i in mod_data]
Al_data = [float(i) for i in Al_data]
R_data = [float(i) for i in R_data]
Q3_data = [float(i) for i in Q3_data]
Q4_data = [float(i) for i in Q4_data]
Q2_data = [float(i) for i in Q2_data]
Q1_data = [float(i) for i in Q1_data]
Q0_data = [float(i) for i in Q0_data]
Al5_data = [float(i) for i in Al5_data]
Al4_data = [float(i) for i in Al4_data]

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

# Model
def model(w1):
    w=np.array([1, abs(w1[0]), abs(w1[1]), abs(w1[2]), abs(w1[3]), abs(w1[4])])
    
    SSE = []
    
# Startværdier
    for i in R_data:    
        
        if i <= 1/3:
            Q3 = [(100/(i+1))-(3*100*i/(i+1)), ]
            Q4 = [0, ]
            Q2 = [3*100*i/(i+1), ]
            Q1 = [0, ]
            Q0 = [0, ]
            Al5 = [100*i/(i+1), ]
            Al4 = [0, ]
        elif 1/3 < i <= 2/3:
            Q3 = [0, ]
            Q4 = [0, ]
            Q2 = [(2*(100*i/(i+1))-3*(100*i/(i+1))*i)/i, ]
            Q1 = [(3*(100*i/(i+1))*i-(100*i/(i+1)))/i, ]
            Q0 = [0, ]
            Al5 = [100*i/(i+1), ]
            Al4 = [0, ]
        elif 2/3 < i <= 1:
            Q3 = [0, ]
            Q4 = [0, ]
            Q2 = [0, ]
            Q1 = [(3*(100*i/(i+1))-3*(100*i/(i+1))*i)/i, ]
            Q0 = [(3*(100*i/(i+1))*i-2*(100*i/(i+1)))/i, ]
            Al5 = [100*i/(i+1), ]
            Al4 = [0, ]
        else:
            continue

        R_ind = R_data.index(i)
    
    #For loop for udregning af p og Q
        for i in draw_ar:
    
            if M2O[i] < (100-M2O[i])*w[5]+(R_data[R_ind]/(1+R_data[R_ind]+(M2O[i]*(1+R_data[R_ind]))/(100-M2O[i])))*100:     
                P = 1
            else:
                P = 0
            
            Ra =  Al5[-1]/(Q3[-1]+ Q2[-1] + Q1[-1] + Q0[-1])
    
            if Ra <= 1/3:
                pQ3 = (Q3[-1]*w[0]) / (Q3[-1]*w[0]+Q4[-1]*w[1]+Q2[-1]*w[2]+Q1[-1]*w[3]+Al5[-1]*w[4])
                pQ4 = (Q4[-1]*w[1]) / (Q3[-1]*w[0]+Q4[-1]*w[1]+Q2[-1]*w[2]+Q1[-1]*w[3]+Al5[-1]*w[4])
                pQ2 = (Q2[-1]*w[2]) / (Q3[-1]*w[0]+Q4[-1]*w[1]+Q2[-1]*w[2]+Q1[-1]*w[3]+Al5[-1]*w[4])
                pQ1 = (Q1[-1]*w[3]) / (Q3[-1]*w[0]+Q4[-1]*w[1]+Q2[-1]*w[2]+Q1[-1]*w[3]+Al5[-1]*w[4])
                pAl5 = (Al5[-1]*w[4]) / (Q3[-1]*w[0]+Q4[-1]*w[1]+Q2[-1]*w[2]+Q1[-1]*w[3]+Al5[-1]*w[4])

                if Q3[-1] - pQ3 - pQ4*pQ3 + 3*pAl5 + 3*pAl5*pQ4 < 0:
                    next_Q3 = 0
                else:
                    next_Q3 = Q3[-1] - pQ3 - pQ4*pQ3 + 3*pAl5 + 3*pAl5*pQ4
                
                if Q4[-1] + pQ3*P - pQ4 + pQ4*pQ3*P - pQ4**2 < 0:
                    next_Q4 = 0
                else:
                    next_Q4 = Q4[-1] + pQ3*P - pQ4 + pQ4*pQ3*P - pQ4**2
                
                if Q2[-1] + pQ4 - pQ2 + pQ3*(1-P) - pQ4*pQ2 + pQ4*pQ3*(1-P) + pQ4**2 - 3*pAl5 - 3*pAl5*pQ4 < 0:
                    next_Q2 = 0
                else:
                    next_Q2 = Q2[-1] + pQ4 - pQ2 + pQ3*(1-P) - pQ4*pQ2 + pQ4*pQ3*(1-P) + pQ4**2 - 3*pAl5 - 3*pAl5*pQ4
                
                if Q1[-1] + pQ2 - pQ1 + pQ4*pQ2 - pQ4*pQ1 < 0:
                    next_Q1 = 0
                else: 
                    next_Q1 = Q1[-1] + pQ2 - pQ1 + pQ4*pQ2 - pQ4*pQ1
                
                if Q0[-1] + pQ1 + pQ4*pQ1 < 0:
                    next_Q0 = 0
                else:
                    next_Q0 = Q0[-1] + pQ1 + pQ4*pQ1
                
                if Al5[-1] - pAl5 - pQ4*pAl5 < 0:
                    next_Al5 = 0
                else: 
                    next_Al5 = Al5[-1] - pAl5 - pQ4*pAl5
                
                if Al4[-1] + pAl5 + pQ4*pAl5 < 0:
                    next_Al4 = 0
                else: 
                    next_Al4 = Al4[-1] + pAl5 + pQ4*pAl5
    
            elif 1/3 < Ra <= 2/3:
                pQ3 = (Q3[-1]*w[0]) / (Q3[-1]*w[0]+Q4[-1]*w[1]+Al5[-1]*w[4])
                pQ4 = (Q4[-1]*w[1]) / (Q3[-1]*w[0]+Q4[-1]*w[1]+Al5[-1]*w[4])
                pAl5 = (Al5[-1]*w[4]) / (Q3[-1]*w[0]+Q4[-1]*w[1]+Al5[-1]*w[4])
                
                if Q3[-1] - pQ3 - pQ4*pQ3 < 0:
                    next_Q3 = 0
                else:
                    next_Q3 = Q3[-1] - pQ3 - pQ4*pQ3
                
                if Q4[-1] + pQ3*P - pQ4 + pQ4*pQ3*P - pQ4**2 < 0:
                    next_Q4 = 0
                else:
                    next_Q4 = Q4[-1] + pQ3*P - pQ4 + pQ4*pQ3*P - pQ4**2
                
                if Q2[-1] + pQ4 + pQ3*(1-P) + pQ4*pQ3*(1-P) + pQ4**2 + 3*pAl5 + 3*pAl5*pQ4 < 0:
                    next_Q2 = 0
                else:
                    next_Q2 = Q2[-1] + pQ4 + pQ3*(1-P) + pQ4*pQ3*(1-P) + pQ4**2 + 3*pAl5 + 3*pAl5*pQ4
                
                if Q1[-1] - 3*pAl5 - 3*pAl5*pQ4 < 0:
                    next_Q1 = 0
                else: 
                    next_Q1 = Q1[-1] - 3*pAl5 - 3*pAl5*pQ4
                
                if Q0[-1]  < 0:
                    next_Q0 = 0
                else:
                    next_Q0 = Q0[-1]
                
                if Al5[-1] - pAl5 - pQ4*pAl5 < 0:
                    next_Al5 = 0
                else: 
                    next_Al5 = Al5[-1] - pAl5 - pQ4*pAl5
                
                if Al4[-1] + pAl5 + pQ4*pAl5 < 0:
                    next_Al4 = 0
                else: 
                    next_Al4 = Al4[-1] + pAl5 + pQ4*pAl5
            
            elif 2/3 < Ra <= 1:
                pQ3 = (Q3[-1]*w[0]) / (Q3[-1]*w[0]+Q4[-1]*w[1]+Al5[-1]*w[4])
                pQ4 = (Q4[-1]*w[1]) / (Q3[-1]*w[0]+Q4[-1]*w[1]+Al5[-1]*w[4])
                pAl5 = (Al5[-1]*w[4]) / (Q3[-1]*w[0]+Q4[-1]*w[1]+Al5[-1]*w[4])
                
                if Q3[-1] - pQ3 - pQ4*pQ3 < 0:
                    next_Q3 = 0
                else:
                    next_Q3 = Q3[-1] - pQ3 - pQ4*pQ3
                
                if Q4[-1] + pQ3*P - pQ4 + pQ4*pQ3*P - pQ4**2 < 0:
                    next_Q4 = 0
                else:
                    next_Q4 = Q4[-1] + pQ3*P - pQ4 + pQ4*pQ3*P - pQ4**2
                
                if Q2[-1] + pQ4 + pQ3*(1-P) + pQ4*pQ3*(1-P) + pQ4**2 < 0:
                    next_Q2 = 0
                else:
                    next_Q2 = Q2[-1] + pQ4 + pQ3*(1-P) + pQ4*pQ3*(1-P) + pQ4**2
                
                if Q1[-1] + 3*pAl5 + 3*pAl5*pQ4 < 0:
                    next_Q1 = 0
                else: 
                    next_Q1 = Q1[-1] + 3*pAl5 + 3*pAl5*pQ4
                
                if Q0[-1] - 3*pAl5 - 3*pAl5*pQ4 < 0:
                    next_Q0 = 0
                else:
                    next_Q0 = Q0[-1] - 3*pAl5 - 3*pAl5*pQ4
                
                if Al5[-1] - pAl5 - pQ4*pAl5< 0:
                    next_Al5 = 0
                else: 
                    next_Al5 = Al5[-1] - pAl5 - pQ4*pAl5
                
                if Al4[-1] + pAl5 + pQ4*pAl5 < 0:
                    next_Al4 = 0
                else: 
                    next_Al4 = Al4[-1] + pAl5 + pQ4*pAl5
            else:
                print ('Your data sucks')
            
            Q3.append(next_Q3)
            Q4.append(next_Q4)
            Q2.append(next_Q2)
            Q1.append(next_Q1)
            Q0.append(next_Q0)
            Al5.append(next_Al5)
            Al4.append(next_Al4)
    
        # Vi laver lister over teoretiske værdier udregnet fra modellen
        mod_m = min(M2O, key=lambda x:abs(x-mod_data[R_ind]))
        ind = M2O.index(mod_m)
        Q3_m = Q3[ind]
        Q4_m = Q4[ind]
        Q2_m = Q2[ind]
        Q1_m = Q1[ind]
        Q0_m = Q0[ind]
        Al5_m = Al5[ind]
        Al4_m = Al4[ind]
        
    
        SSSE = (((Q3_m - Q3_data[R_ind])**2)+((Q4_m - Q4_data[R_ind])**2)+((Q2_m - Q2_data[R_ind])**2)+((Q1_m - Q1_data[R_ind])**2)+((Q0_m - Q0_data[R_ind])**2)+((Al5_m - Al5_data[R_ind])**2)+((Al4_m - Al4_data[R_ind])**2))
        SSE.append(SSSE)
    
    return sum(SSE)

SSElist =[]
it = 1
def print_fun(x, f, accepted):
    global it
    global SSElist 
    print("SSE = {} for {}".format(f,it))
    it += 1
    SSElist.append(f)

#w1 = [0.5, 0.6, 0, 0.5, 0.3]

#res = scipy.optimize.basinhopping(model, w1, niter=5000, T=1.0, stepsize=0.5, minimizer_kwargs=None, take_step=None, accept_test=None, callback=print_fun, interval=50, disp=False, niter_success=None)

#w = [1, abs(res.x[0]), abs(res.x[1]), abs(res.x[2]), abs(res.x[3]), abs(res.x[4])]

w = [1,1.0344376953,9.43190731879e-11,4.36394741362e-09,21.0272955758,0.259023946613]

RMSElist = []

for i in SSElist:
    RMSE = math.sqrt(i/(7*len(R_data)))
    RMSElist.append(RMSE)
    
tab = tt.Texttable()
headings = ['W','Na']
tab.header(headings)
names = ['Q3', 'Q4', 'Q2', 'Q1', 'Al5/6', 'A' ]
Na = w
for row in zip(names,Na):
    tab.add_row(row)
s = tab.draw()
print (s)

with open('Results_{}_{}_{}.csv'.format(file_name, fil, tid), 'w', newline = '') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(w)
    for a in list(range(len(SSElist))):
        filewriter.writerow((SSElist[a], RMSElist[a]))

## Herfra plotter vi ##

Q3_m = []
Q4_m = []
Q2_m = []
Q1_m = []
Q0_m = []
Al5_m = []
Al4_m = []

# Startværdier
for i in R_data:    
    
    if i <= 1/3:
        Q3 = [(100/(i+1))-(3*100*i/(i+1)), ]
        Q4 = [0, ]
        Q2 = [3*100*i/(i+1), ]
        Q1 = [0, ]
        Q0 = [0, ]
        Al5 = [100*i/(i+1), ]
        Al4 = [0, ]
    elif 1/3 < i <= 2/3:
        Q3 = [0, ]
        Q4 = [0, ]
        Q2 = [(2*(100*i/(i+1))-3*(100*i/(i+1))*i)/i, ]
        Q1 = [(3*(100*i/(i+1))*i-(100*i/(i+1)))/i, ]
        Q0 = [0, ]
        Al5 = [100*i/(i+1), ]
        Al4 = [0, ]
    elif 2/3 < i <= 1:
        Q3 = [0, ]
        Q4 = [0, ]
        Q2 = [0, ]
        Q1 = [(3*(100*i/(i+1))-3*(100*i/(i+1))*i)/i, ]
        Q0 = [(3*(100*i/(i+1))*i-2*(100*i/(i+1)))/i, ]
        Al5 = [100*i/(i+1), ]
        Al4 = [0, ]
    else:
        continue

    R_ind = R_data.index(i)
    

#For loop for udregning af p og Q
    for i in draw_ar:
        
        if M2O[i] < (100-M2O[i])*w[5]+(R_data[R_ind]/(1+R_data[R_ind]+(M2O[i]*(1+R_data[R_ind]))/(100-M2O[i])))*100:     
            P = 1
        else:
            P = 0
        
        Ra =  Al5[-1]/(Q3[-1]+ Q2[-1] + Q1[-1] + Q0[-1])

        if Ra <= 1/3:
            pQ3 = (Q3[-1]*w[0]) / (Q3[-1]*w[0]+Q4[-1]*w[1]+Q2[-1]*w[2]+Q1[-1]*w[3]+Al5[-1]*w[4])
            pQ4 = (Q4[-1]*w[1]) / (Q3[-1]*w[0]+Q4[-1]*w[1]+Q2[-1]*w[2]+Q1[-1]*w[3]+Al5[-1]*w[4])
            pQ2 = (Q2[-1]*w[2]) / (Q3[-1]*w[0]+Q4[-1]*w[1]+Q2[-1]*w[2]+Q1[-1]*w[3]+Al5[-1]*w[4])
            pQ1 = (Q1[-1]*w[3]) / (Q3[-1]*w[0]+Q4[-1]*w[1]+Q2[-1]*w[2]+Q1[-1]*w[3]+Al5[-1]*w[4])
            pAl5 = (Al5[-1]*w[4]) / (Q3[-1]*w[0]+Q4[-1]*w[1]+Q2[-1]*w[2]+Q1[-1]*w[3]+Al5[-1]*w[4])

            if Q3[-1] - pQ3 - pQ4*pQ3 + 3*pAl5 + 3*pAl5*pQ4 < 0:
                next_Q3 = 0
            else:
                next_Q3 = Q3[-1] - pQ3 - pQ4*pQ3 + 3*pAl5 + 3*pAl5*pQ4
            
            if Q4[-1] + pQ3*P - pQ4 + pQ4*pQ3*P - pQ4**2 < 0:
                next_Q4 = 0
            else:
                next_Q4 = Q4[-1] + pQ3*P - pQ4 + pQ4*pQ3*P - pQ4**2
            
            if Q2[-1] + pQ4 - pQ2 + pQ3*(1-P) - pQ4*pQ2 + pQ4*pQ3*(1-P) + pQ4**2 - 3*pAl5 - 3*pAl5*pQ4 < 0:
                next_Q2 = 0
            else:
                next_Q2 = Q2[-1] + pQ4 - pQ2 + pQ3*(1-P) - pQ4*pQ2 + pQ4*pQ3*(1-P) + pQ4**2 - 3*pAl5 - 3*pAl5*pQ4
            
            if Q1[-1] + pQ2 - pQ1 + pQ4*pQ2 - pQ4*pQ1 < 0:
                next_Q1 = 0
            else: 
                next_Q1 = Q1[-1] + pQ2 - pQ1 + pQ4*pQ2 - pQ4*pQ1
            
            if Q0[-1] + pQ1 + pQ4*pQ1 < 0:
                next_Q0 = 0
            else:
                next_Q0 = Q0[-1] + pQ1 + pQ4*pQ1
            
            if Al5[-1] - pAl5 - pQ4*pAl5 < 0:
                next_Al5 = 0
            else: 
                next_Al5 = Al5[-1] - pAl5 - pQ4*pAl5
            
            if Al4[-1] + pAl5 + pQ4*pAl5 < 0:
                next_Al4 = 0
            else: 
                next_Al4 = Al4[-1] + pAl5 + pQ4*pAl5

        elif 1/3 < Ra <= 2/3:
            pQ3 = (Q3[-1]*w[0]) / (Q3[-1]*w[0]+Q4[-1]*w[1]+Al5[-1]*w[4])
            pQ4 = (Q4[-1]*w[1]) / (Q3[-1]*w[0]+Q4[-1]*w[1]+Al5[-1]*w[4])
            pAl5 = (Al5[-1]*w[4]) / (Q3[-1]*w[0]+Q4[-1]*w[1]+Al5[-1]*w[4])
            
            if Q3[-1] - pQ3 - pQ4*pQ3 < 0:
                next_Q3 = 0
            else:
                next_Q3 = Q3[-1] - pQ3 - pQ4*pQ3
            
            if Q4[-1] + pQ3*P - pQ4 + pQ4*pQ3*P - pQ4**2 < 0:
                next_Q4 = 0
            else:
                next_Q4 = Q4[-1] + pQ3*P - pQ4 + pQ4*pQ3*P - pQ4**2
            
            if Q2[-1] + pQ4 + pQ3*(1-P) + pQ4*pQ3*(1-P) + pQ4**2 + 3*pAl5 + 3*pAl5*pQ4 < 0:
                next_Q2 = 0
            else:
                next_Q2 = Q2[-1] + pQ4 + pQ3*(1-P) + pQ4*pQ3*(1-P) + pQ4**2 + 3*pAl5 + 3*pAl5*pQ4
            
            if Q1[-1] - 3*pAl5 - 3*pAl5*pQ4 < 0:
                next_Q1 = 0
            else: 
                next_Q1 = Q1[-1] - 3*pAl5 - 3*pAl5*pQ4
            
            if Q0[-1]  < 0:
                next_Q0 = 0
            else:
                next_Q0 = Q0[-1]
            
            if Al5[-1] - pAl5 - pQ4*pAl5 < 0:
                next_Al5 = 0
            else: 
                next_Al5 = Al5[-1] - pAl5 - pQ4*pAl5
            
            if Al4[-1] + pAl5 + pQ4*pAl5 < 0:
                next_Al4 = 0
            else: 
                next_Al4 = Al4[-1] + pAl5 + pQ4*pAl5
        
        elif 2/3 < Ra <= 1:
            pQ3 = (Q3[-1]*w[0]) / (Q3[-1]*w[0]+Q4[-1]*w[1]+Al5[-1]*w[4])
            pQ4 = (Q4[-1]*w[1]) / (Q3[-1]*w[0]+Q4[-1]*w[1]+Al5[-1]*w[4])
            pAl5 = (Al5[-1]*w[4]) / (Q3[-1]*w[0]+Q4[-1]*w[1]+Al5[-1]*w[4])
            
            if Q3[-1] - pQ3 - pQ4*pQ3 < 0:
                next_Q3 = 0
            else:
                next_Q3 = Q3[-1] - pQ3 - pQ4*pQ3
            
            if Q4[-1] + pQ3*P - pQ4 + pQ4*pQ3*P - pQ4**2 < 0:
                next_Q4 = 0
            else:
                next_Q4 = Q4[-1] + pQ3*P - pQ4 + pQ4*pQ3*P - pQ4**2
            
            if Q2[-1] + pQ4 + pQ3*(1-P) + pQ4*pQ3*(1-P) + pQ4**2 < 0:
                next_Q2 = 0
            else:
                next_Q2 = Q2[-1] + pQ4 + pQ3*(1-P) + pQ4*pQ3*(1-P) + pQ4**2
            
            if Q1[-1] + 3*pAl5 + 3*pAl5*pQ4 < 0:
                next_Q1 = 0
            else: 
                next_Q1 = Q1[-1] + 3*pAl5 + 3*pAl5*pQ4
            
            if Q0[-1] - 3*pAl5 - 3*pAl5*pQ4 < 0:
                next_Q0 = 0
            else:
                next_Q0 = Q0[-1] - 3*pAl5 - 3*pAl5*pQ4
            
            if Al5[-1] - pAl5 - pQ4*pAl5< 0:
                next_Al5 = 0
            else: 
                next_Al5 = Al5[-1] - pAl5 - pQ4*pAl5
            
            if Al4[-1] + pAl5 + pQ4*pAl5 < 0:
                next_Al4 = 0
            else: 
                next_Al4 = Al4[-1] + pAl5 + pQ4*pAl5
        else:
            print ('Your data sucks')
            
        Q3.append(next_Q3)
        Q4.append(next_Q4)
        Q2.append(next_Q2)
        Q1.append(next_Q1)
        Q0.append(next_Q0)
        Al5.append(next_Al5)
        Al4.append(next_Al4)

    # Vi laver lister over teoretiske værdier udregnet fra modellen
    mod_m = min(M2O, key=lambda x:abs(x-mod_data[R_ind]))
    ind = M2O.index(mod_m)
    Q3_m.append(Q3[ind])
    Q4_m.append(Q4[ind])
    Q2_m.append(Q2[ind])
    Q1_m.append(Q1[ind])
    Q0_m.append(Q0[ind])
    Al5_m.append(Al5[ind])
    Al4_m.append(Al4[ind])
    
    if R_ind == 3:
        fig1 = plt.figure(figsize=(12, 7))
        plt.subplot(111)
        plt.plot(M2O, Q3, 'r', M2O, Q4, 'y', M2O, Q2, 'g', M2O, Al4, 'm', M2O, Al5, 'b', M2O, Q1, 'c', M2O, Q0, 'k', mod_data[R_ind], Q3_data[R_ind], 'rd', mod_data[R_ind], Q4_data[R_ind], 'yd', mod_data[R_ind], Q2_data[R_ind], 'gd', mod_data[R_ind], Al4_data[R_ind], 'md', mod_data[R_ind], Al5_data[R_ind], 'bd')
        plt.xlabel("M2O (mol %)")
        plt.ylabel("Qn/Al distribution (mol %)")
        plt.title('Distribution')
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)

        plt.savefig("{}_modifier_{}_{}.svg".format(file_name, fil, tid))
    
    
Q3_m = np.array(Q3_m)
Q4_m = np.array(Q4_m)
Q2_m = np.array(Q2_m)
Q1_m = np.array(Q1_m)
Q0_m = np.array(Q0_m)
Al5_m = np.array(Al5_m)
Al4_m = np.array(Al4_m)

SSE = sum(((Q4_data - Q4_m)**2) + ((Q3_data - Q3_m)**2) + ((Q2_data - Q2_m)**2) + ((Q1_data - Q1_m)**2) + ((Q0_data - Q0_m)**2) + ((Al4_data - Al4_m)**2) + ((Al5_data - Al5_m)**2))
print(SSE)
print(math.sqrt(SSE/(7*len(R_data))))

with open('Model_data_{}_{}_{}.csv'.format(file_name, fil, tid), 'w', newline = '') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for a in list(range(len(Q3_m))):
        filewriter.writerow((Q3_m[a], Q4_m[a], Q2_m[a], Q1_m[a], Q0_m[a], Al5_m[a], Al4_m[a]))
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


if fil == "Kapoor_2018":
    
    pltAl = [Al_data[0], Al_data[1], Al_data[2], Al_data[3], Al_data[4], Al_data[5]]
    pltNa = [mod_data[6], mod_data[7], mod_data[8], mod_data[9], mod_data[10], mod_data[11], mod_data[12]]
    
    Al203 = np.array(pltAl)
    Na2O = np.array(pltNa)
    
    fig = plt.figure(figsize=(14, 7))
    
    plt.subplot(131)
    b3, b4, b2, b1, b0, a4, a5 = plt.plot(Q3_data, Q3_m,'ro', Q4_data, Q4_m , 'ys', Q2_data, Q2_m, 'gd', Q1_data , Q1_m, 'cp', Q0_data, Q0_m, 'k*', Al4_data, Al4_m , 'm^', Al5_data, Al5_m , 'bv')
    plt.plot(a ,fit_fn(a),':k')
    plt.xlabel("Experimental (%)", fontproperties=font)
    plt.ylabel("Model (%)", fontproperties=font)
    #plt.legend(handles=[Q3_red, Q4_yellow, Q2_green, Al5_blue, Al4_magenta, Q1_cyan, Q0_black])
    
    plt.subplot(132)
    plt.plot(Al203, Q3_m[0:6], 'r', Al203, Q4_m[0:6], 'y', Al203, Q2_m[0:6], 'g', Al203, Al4_m[0:6], 'm', Al203, Al5_m[0:6], 'b', Al203, Q1_m[0:6], 'c', Al203, Q0_m[0:6], 'k',
             Al203, Q3_data[0:6], 'ro', Al203, Q4_data[0:6], 'ys', Al203, Q2_data[0:6], 'gd', Al203, Al4_data[0:6], 'm^', Al203, Al5_data[0:6], 'bv')
    plt.xlabel("Al$_2$O$_3$ (mol%)", fontproperties=font)
    plt.ylabel("Species fraction (%)", fontproperties=font)
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                        wspace=0.35)
    
    plt.subplot(133)
    plt.plot(Na2O, Q3_m[6:13], 'r', Na2O, Q4_m[6:13], 'y', Na2O, Q2_m[6:13], 'g', Na2O, Al4_m[6:13], 'm', Na2O, Al5_m[6:13], 'b', Na2O, Q1_m[6:13], 'c', Na2O, Q0_m[6:13], 'k',
             Na2O, Q3_data[6:13], 'ro', Na2O, Q4_data[6:13], 'ys', Na2O, Q2_data[6:13], 'gd', Na2O, Al4_data[6:13], 'm^', Na2O, Al5_data[6:13], 'bv')
    plt.xlabel("Na$_2$O (mol%)", fontproperties=font)
    plt.ylabel("Species fraction (%)", fontproperties=font)
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.91, hspace=0.25,
                        wspace=0.35)
    fig.legend((b3, b4, b2, b1, b0, a5, a4), ('Q$^3$', 'Q$^4$', 'Q$^2$', 'Q$^1$','Q$^0$', 'Al$^5$', 'Al$^4$'), loc = (0.92, 0.4))
    plt.savefig("{}_{}_{}.svg".format(file_name, fil, tid))
else:
    fig = plt.figure(figsize=(7.5, 7))
    
    b3, b4, b2, b1, b0, a4, a5 = plt.plot(Q3_data, Q3_m, 'ro', Q4_data, Q4_m , 'ys', Q2_data, Q2_m, 'gd', Q1_data , Q1_m, 'cp', Q0_data, Q0_m, 'k*', Al4_data, Al4_m , 'm^', Al5_data, Al5_m , 'bv' )
    plt.plot(a ,fit_fn(a),':k')
    plt.xlabel("Experimental (%)", fontproperties=font)
    plt.ylabel("Model (%)", fontproperties=font)
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.85, hspace=0.25,
                        wspace=0.35)
    fig.legend((b3, b4, b2, b1, b0, a5, a4), ('Q$^3$', 'Q$^4$', 'Q$^2$', 'Q$^1$','Q$^0$', 'Al$^5$', 'Al$^4$'), loc = (0.86, 0.4))
    plt.savefig("{}_{}_{}.svg".format(file_name, fil, tid))
