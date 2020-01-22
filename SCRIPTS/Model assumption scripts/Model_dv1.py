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

# Model
def model(w1):
    w=np.array([1, abs(w1[0]), abs(w1[1]), abs(w1[2]), abs(w1[3]), abs(w1[4]), abs(w1[5]), abs(w1[6]), abs(w1[7]), abs(w1[8]), abs(w1[9])])

    SSE = []

# Startværdier
    for i in R_data:
        Q3 = [(100/(i+1)), ]
        Q4 = [0, ]
        Q2 = [0, ]
        Q1 = [0, ]
        Q0 = [0, ]
        Al5 = [100*i/(i+1), ]
        Al4 = [0, ]

        R_ind = R_data.index(i)

        Q3_c = [ ]
        Q4_c = [ ]
        Q2_c = [ ]
        Q1_c = [ ]
        Q0_c = [ ]
        Al5_c = [ ]
        Al4_c = [ ]


        #For loop for udregning af g og Q_Al5
        for i in draw_ar:

            Q3_Al5 = [Q3[-1], ]
            Q4_Al5 = [Q4[-1],  ]
            Q2_Al5 = [Q2[-1],  ]
            Q1_Al5 = [Q1[-1],  ]
            Q0_Al5 = [Q0[-1],  ]
            Al5_Al5 = [Al5[-1],  ]
            Al4_Al5 = [Al4[-1],  ]

            #Q4
            Q4A_Al5 = [0, ]
            Q4N_Al5 = [Q4[-1], ]

            #Q2
            Q2A_Al5 = [0, ]
            Q2N_Al5 = [Q2[-1], ]

            #Q1
            Q1NN_Al5 = [Q1[-1], ]
            Q1AN_Al5 = [0, ]
            Q1AA_Al5 = [0, ]

            #Q0
            Q0AAA_Al5 = [0, ]
            Q0AAN_Al5 = [0, ]
            Q0ANN_Al5 = [0, ]
            Q0NNN_Al5 = [Q0[-1], ]

            #Al draw
            Al_draw = [0, ]

            while (Al_draw[-1]/3)*100/(100+(Al_draw[-1]/3)) < Al5_Al5[-1]*100/(100-Al4_Al5[-1]):
                gQ3 = (Q3_Al5[-1]*w[6]) / (Q3_Al5[-1]*w[6]+Q4_Al5[-1]*w[7]+Q2_Al5[-1]*w[8]+Q1_Al5[-1]*w[9]+Al5_Al5[-1]*w[10])
                gQ4 = (Q4_Al5[-1]*w[7]) / (Q3_Al5[-1]*w[6]+Q4_Al5[-1]*w[7]+Q2_Al5[-1]*w[8]+Q1_Al5[-1]*w[9]+Al5_Al5[-1]*w[10])
                gQ2 = (Q2_Al5[-1]*w[8]) / (Q3_Al5[-1]*w[6]+Q4_Al5[-1]*w[7]+Q2_Al5[-1]*w[8]+Q1_Al5[-1]*w[9]+Al5_Al5[-1]*w[10])
                gQ1 = (Q1_Al5[-1]*w[9]) / (Q3_Al5[-1]*w[6]+Q4_Al5[-1]*w[7]+Q2_Al5[-1]*w[8]+Q1_Al5[-1]*w[9]+Al5_Al5[-1]*w[10])
                gAl5 = (Al5_Al5[-1]*w[10]) / (Q3_Al5[-1]*w[6]+Q4_Al5[-1]*w[7]+Q2_Al5[-1]*w[8]+Q1_Al5[-1]*w[9]+Al5_Al5[-1]*w[10])


                if Q4A_Al5[-1]+Q4N_Al5[-1] > 0:
                    rQ4A = Q4A_Al5[-1]/(Q4A_Al5[-1]+Q4N_Al5[-1])
                    rQ4N = Q4N_Al5[-1]/(Q4A_Al5[-1]+Q4N_Al5[-1])
                else:
                    rQ4A = 0
                    rQ4N = 0

                if Q2A_Al5[-1]+Q2N_Al5[-1] > 0:
                    rQ2A = Q2A_Al5[-1]/(Q2A_Al5[-1]+Q2N_Al5[-1])
                    rQ2N = Q2N_Al5[-1]/(Q2A_Al5[-1]+Q2N_Al5[-1])
                else:
                    rQ2A = 0
                    rQ2N = 0

                if Q1AA_Al5[-1]+Q1AN_Al5[-1]+Q1NN_Al5[-1] > 0:
                    rQ1AA = Q1AA_Al5[-1]/(Q1AA_Al5[-1]+Q1AN_Al5[-1]+Q1NN_Al5[-1])
                    rQ1AN = Q1AN_Al5[-1]/(Q1AA_Al5[-1]+Q1AN_Al5[-1]+Q1NN_Al5[-1])
                    rQ1NN = Q1NN_Al5[-1]/(Q1AA_Al5[-1]+Q1AN_Al5[-1]+Q1NN_Al5[-1])
                else:
                    rQ1AA = 0
                    rQ1AN = 0
                    rQ1NN = 0

                #overgang
                if (Q4_Al5[-1] + Q2_Al5[-1] + 2*Q1_Al5[-1] + 3*Q0_Al5[-1]) < Q3[0]*w[5]:
                    P = 1
                else:
                    P = 0

                # Draws
                if Q3_Al5[-1] - gQ3 + (- gQ3)*gQ4 > 0:
                    next_Q3_Al5 = Q3_Al5[-1] - gQ3 + (- gQ3)*gQ4

                else:
                    next_Q3_Al5 = 0

                if Q4_Al5[-1] + gQ3*P - gQ4 + (gQ3*P - gQ4)*gQ4 > 0:
                    next_Q4A_Al5 = Q4A_Al5[-1] + gQ3*P - gQ4*rQ4A + (gQ3*P - gQ4*rQ4A)*gQ4
                    next_Q4N_Al5 = Q4N_Al5[-1] - gQ4*rQ4N - (gQ4*rQ4N)*gQ4
                    next_Q4_Al5 = next_Q4A_Al5 + next_Q4N_Al5
                else:
                    next_Q4A_Al5 = 0
                    next_Q4N_Al5 = 0
                    next_Q4_Al5 = 0

                if Q2_Al5[-1] + gQ4 + gQ3*(1-P) - gQ2 + (gQ4 + gQ3*(1-P) - gQ2)*gQ4 > 0:
                    next_Q2A_Al5 = Q2A_Al5[-1] + gQ4*rQ4A + gQ3*(1-P) - gQ2*rQ2A + (gQ4*rQ4A + gQ3*(1-P) - gQ2*rQ2A)*gQ4
                    next_Q2N_Al5 = Q2N_Al5[-1] + gQ4*rQ4N - gQ2*rQ2N + (gQ4*rQ4N - gQ2*rQ2N)*gQ4
                    next_Q2_Al5 = next_Q2A_Al5 + next_Q2N_Al5
                else:
                    next_Q2A_Al5 = 0
                    next_Q2N_Al5 = 0
                    next_Q2_Al5 = 0

                if Q1_Al5[-1] + gQ2 - gQ1 + (gQ2 - gQ1)*gQ4 < 0:
                    next_Q1AA_Al5 = 0
                    next_Q1AN_Al5 = 0
                    next_Q1NN_Al5 = 0
                    next_Q1_Al5 = 0
                else:
                    next_Q1AA_Al5 = Q1AA_Al5[-1] + gQ2*rQ2A - gQ1*rQ1AA + (gQ2*rQ2A - gQ1*rQ1AA)*gQ4
                    next_Q1AN_Al5 = Q1AN_Al5[-1] + gQ2*rQ2N - gQ1*rQ1AN + (gQ2*rQ2N - gQ1*rQ1AN)*gQ4
                    next_Q1NN_Al5 = Q1NN_Al5[-1] - gQ1*rQ1NN - (gQ1*rQ1NN)*gQ4
                    next_Q1_Al5 = next_Q1AA_Al5 + next_Q1AN_Al5 + next_Q1NN_Al5

                if Q0_Al5[-1] + gQ1 + (gQ1)*gQ4 < 0:
                    next_Q0AAA_Al5 = 0
                    next_Q0AAN_Al5 = 0
                    next_Q0ANN_Al5 = 0
                    next_Q1_Al5 = 0
                else:
                    next_Q0AAA_Al5 = Q0AAA_Al5[-1] + gQ1*rQ1AA + (gQ1*rQ1AA)*gQ4
                    next_Q0AAN_Al5 = Q0AAN_Al5[-1] + gQ1*rQ1AN + (gQ1*rQ1AN)*gQ4
                    next_Q0ANN_Al5 = Q0ANN_Al5[-1] + gQ1*rQ1NN + (gQ1*rQ1NN)*gQ4
                    next_Q0_Al5 = next_Q0AAA_Al5 + next_Q0AAN_Al5 + next_Q0ANN_Al5 + Q0NNN_Al5[-1]

                if Al5_Al5[-1] - gAl5 + (-gAl5)*gQ4 > 0:
                    next_Al5_Al5 = Al5_Al5[-1] - gAl5 + (-gAl5)*gQ4
                else:
                    next_Al5_Al5 = 0

                if Al4_Al5[-1] + gAl5 + (gAl5)*gQ4 > 0:
                    next_Al4_Al5 = Al4_Al5[-1] + gAl5 + (gAl5)*gQ4
                else:
                    next_Al4_Al5 = 0

                next_Al_draw = Al_draw[-1]+1
                Al_draw.append(next_Al_draw)

                Q3_Al5.append(next_Q3_Al5)

                Q4_Al5.append(next_Q4_Al5)
                Q4A_Al5.append(next_Q4A_Al5)
                Q4N_Al5.append(next_Q4N_Al5)

                Q2_Al5.append(next_Q2_Al5)
                Q2A_Al5.append(next_Q2A_Al5)
                Q2N_Al5.append(next_Q2N_Al5)

                Q1_Al5.append(next_Q1_Al5)
                Q1AA_Al5.append(next_Q1AA_Al5)
                Q1AN_Al5.append(next_Q1AN_Al5)
                Q1NN_Al5.append(next_Q1NN_Al5)

                Q0_Al5.append(next_Q0_Al5)
                Q0AAA_Al5.append(next_Q0AAA_Al5)
                Q0AAN_Al5.append(next_Q0AAN_Al5)
                Q0ANN_Al5.append(next_Q0ANN_Al5)

                Al5_Al5.append(next_Al5_Al5)
                Al4_Al5.append(next_Al4_Al5)

            ## Residual Al draw ##
            if Al4_Al5[-1] + Al5_Al5[-1] > 0:
                aa = ((-300*(Al5_Al5[-1]*100/(100-Al4_Al5[-1])))/(-100+(Al5_Al5[-1]*100/(100-Al4_Al5[-1]))))-(-300*(Al5_Al5[-2]*100/(100-Al4_Al5[-2])))/(-100+(Al5_Al5[-2]*100/(100-Al4_Al5[-2])))
                b = -aa*Al_draw[-1]+(-300*(Al5_Al5[-1]*100/(100-Al4_Al5[-1])))/(-100+(Al5_Al5[-1]*100/(100-Al4_Al5[-1])))
                balance_draw = (-b/(aa-1))-Al_draw[-2]
            else:
                balance_draw = 0

            gQ3 = balance_draw*(Q3_Al5[-1]*w[6]) / (Q3_Al5[-1]*w[6]+Q4_Al5[-1]*w[7]+Q2_Al5[-1]*w[8]+Q1_Al5[-1]*w[9]+Al5_Al5[-1]*w[10])
            gQ4 = balance_draw*(Q4_Al5[-1]*w[7]) / (Q3_Al5[-1]*w[6]+Q4_Al5[-1]*w[7]+Q2_Al5[-1]*w[8]+Q1_Al5[-1]*w[9]+Al5_Al5[-1]*w[10])
            gQ2 = balance_draw*(Q2_Al5[-1]*w[8]) / (Q3_Al5[-1]*w[6]+Q4_Al5[-1]*w[7]+Q2_Al5[-1]*w[8]+Q1_Al5[-1]*w[9]+Al5_Al5[-1]*w[10])
            gQ1 = balance_draw*(Q1_Al5[-1]*w[9]) / (Q3_Al5[-1]*w[6]+Q4_Al5[-1]*w[7]+Q2_Al5[-1]*w[8]+Q1_Al5[-1]*w[9]+Al5_Al5[-1]*w[10])
            gAl5 = balance_draw*(Al5_Al5[-1]*w[10]) / (Q3_Al5[-1]*w[6]+Q4_Al5[-1]*w[7]+Q2_Al5[-1]*w[8]+Q1_Al5[-1]*w[9]+Al5_Al5[-1]*w[10])

            if Q4A_Al5[-1]+Q4N_Al5[-1] > 0:
                rQ4A = Q4A_Al5[-1]/(Q4A_Al5[-1]+Q4N_Al5[-1])
                rQ4N = Q4N_Al5[-1]/(Q4A_Al5[-1]+Q4N_Al5[-1])
            else:
                rQ4A = 0
                rQ4N = 0

            if Q2A_Al5[-1]+Q2N_Al5[-1] > 0:
                rQ2A = Q2A_Al5[-1]/(Q2A_Al5[-1]+Q2N_Al5[-1])
                rQ2N = Q2N_Al5[-1]/(Q2A_Al5[-1]+Q2N_Al5[-1])
            else:
                rQ2A = 0
                rQ2N = 0

            if Q1AA_Al5[-1]+Q1AN_Al5[-1]+Q1NN_Al5[-1] > 0:
                rQ1AA = Q1AA_Al5[-1]/(Q1AA_Al5[-1]+Q1AN_Al5[-1]+Q1NN_Al5[-1])
                rQ1AN = Q1AN_Al5[-1]/(Q1AA_Al5[-1]+Q1AN_Al5[-1]+Q1NN_Al5[-1])
                rQ1NN = Q1NN_Al5[-1]/(Q1AA_Al5[-1]+Q1AN_Al5[-1]+Q1NN_Al5[-1])
            else:
                rQ1AA = 0
                rQ1AN = 0
                rQ1NN = 0

              #overgang
            if (Q4_Al5[-1] + Q2_Al5[-1] + 2*Q1_Al5[-1] + 3*Q0_Al5[-1]) < Q3[0]*w[5]:
                P = 1
            else:
                P = 0

                # Draws

            if Q3_Al5[-1] - gQ3 + (- gQ3)*gQ4 > 0:
                next_Q3_Al5 = Q3_Al5[-1] - gQ3 + (- gQ3)*gQ4

            else:
                next_Q3_Al5 = 0

            if Q4_Al5[-1] + gQ3*P - gQ4 + (gQ3*P - gQ4)*gQ4 > 0:
                next_Q4A_Al5 = Q4A_Al5[-1] + gQ3*P - gQ4*rQ4A + (gQ3*P - gQ4*rQ4A)*gQ4
                next_Q4N_Al5 = Q4N_Al5[-1] - gQ4*rQ4N - (gQ4*rQ4N)*gQ4
                next_Q4_Al5 = next_Q4A_Al5 + next_Q4N_Al5
            else:
                next_Q4A_Al5 = 0
                next_Q4N_Al5 = 0
                next_Q4_Al5 = 0

            if Q2_Al5[-1] + gQ4 + gQ3*(1-P) - gQ2 + (gQ4 + gQ3*(1-P) - gQ2)*gQ4 > 0:
                next_Q2A_Al5 = Q2A_Al5[-1] + gQ4*rQ4A + gQ3*(1-P) - gQ2*rQ2A + (gQ4*rQ4A + gQ3*(1-P) - gQ2*rQ2A)*gQ4
                next_Q2N_Al5 = Q2N_Al5[-1] + gQ4*rQ4N - gQ2*rQ2N + (gQ4*rQ4N - gQ2*rQ2N)*gQ4
                next_Q2_Al5 = next_Q2A_Al5 + next_Q2N_Al5
            else:
                next_Q2A_Al5 = 0
                next_Q2N_Al5 = 0
                next_Q2_Al5 = 0

            if Q1_Al5[-1] + gQ2 - gQ1 + (gQ2 - gQ1)*gQ4 < 0:
                next_Q1AA_Al5 = 0
                next_Q1AN_Al5 = 0
                next_Q1NN_Al5 = 0
                next_Q1_Al5 = 0
            else:
                next_Q1AA_Al5 = Q1AA_Al5[-1] + gQ2*rQ2A - gQ1*rQ1AA + (gQ2*rQ2A - gQ1*rQ1AA)*gQ4
                next_Q1AN_Al5 = Q1AN_Al5[-1] + gQ2*rQ2N - gQ1*rQ1AN + (gQ2*rQ2N - gQ1*rQ1AN)*gQ4
                next_Q1NN_Al5 = Q1NN_Al5[-1] - gQ1*rQ1NN - (gQ1*rQ1NN)*gQ4
                next_Q1_Al5 = next_Q1AA_Al5 + next_Q1AN_Al5 + next_Q1NN_Al5

            if Q0_Al5[-1] + gQ1 + (gQ1)*gQ4 < 0:
                next_Q0AAA_Al5 = 0
                next_Q0AAN_Al5 = 0
                next_Q0ANN_Al5 = 0
                next_Q1_Al5 = 0
            else:
                next_Q0AAA_Al5 = Q0AAA_Al5[-1] + gQ1*rQ1AA + (gQ1*rQ1AA)*gQ4
                next_Q0AAN_Al5 = Q0AAN_Al5[-1] + gQ1*rQ1AN + (gQ1*rQ1AN)*gQ4
                next_Q0ANN_Al5 = Q0ANN_Al5[-1] + gQ1*rQ1NN + (gQ1*rQ1NN)*gQ4
                next_Q0_Al5 = next_Q0AAA_Al5 + next_Q0AAN_Al5 + next_Q0ANN_Al5 + Q0NNN_Al5[-1]

            if Al5_Al5[-1] - gAl5 + (-gAl5)*gQ4 > 0:
                next_Al5_Al5 = Al5_Al5[-1] - gAl5 + (-gAl5)*gQ4
            else:
                next_Al5_Al5 = 0

            if Al4_Al5[-1] + gAl5 + (gAl5)*gQ4 > 0:
                next_Al4_Al5 = Al4_Al5[-1] + gAl5 + (gAl5)*gQ4
            else:
                next_Al4_Al5 = 0

            Q3_Al5.append(next_Q3_Al5)

            Q4_Al5.append(next_Q4_Al5)
            Q4A_Al5.append(next_Q4A_Al5)
            Q4N_Al5.append(next_Q4N_Al5)

            Q2_Al5.append(next_Q2_Al5)
            Q2A_Al5.append(next_Q2A_Al5)
            Q2N_Al5.append(next_Q2N_Al5)

            Q1_Al5.append(next_Q1_Al5)
            Q1AA_Al5.append(next_Q1AA_Al5)
            Q1AN_Al5.append(next_Q1AN_Al5)
            Q1NN_Al5.append(next_Q1NN_Al5)

            Q0_Al5.append(next_Q0_Al5)
            Q0AAA_Al5.append(next_Q0AAA_Al5)
            Q0AAN_Al5.append(next_Q0AAN_Al5)
            Q0ANN_Al5.append(next_Q0ANN_Al5)

            Al5_Al5.append(next_Al5_Al5)
            Al4_Al5.append(next_Al4_Al5)


            ## M2O DRAW ##
            pQ3 = (Q3_Al5[-1]*w[0]) / (Q3_Al5[-1]*w[0]+Q4_Al5[-1]*w[1]+Q2_Al5[-1]*w[2]+Q1_Al5[-1]*w[3]+Al5_Al5[-1]*w[4])
            pQ4 = (Q4_Al5[-1]*w[1]) / (Q3_Al5[-1]*w[0]+Q4_Al5[-1]*w[1]+Q2_Al5[-1]*w[2]+Q1_Al5[-1]*w[3]+Al5_Al5[-1]*w[4])
            pQ2 = (Q2_Al5[-1]*w[2]) / (Q3_Al5[-1]*w[0]+Q4_Al5[-1]*w[1]+Q2_Al5[-1]*w[2]+Q1_Al5[-1]*w[3]+Al5_Al5[-1]*w[4])
            pQ1 = (Q1_Al5[-1]*w[3]) / (Q3_Al5[-1]*w[0]+Q4_Al5[-1]*w[1]+Q2_Al5[-1]*w[2]+Q1_Al5[-1]*w[3]+Al5_Al5[-1]*w[4])
            pAl5 = (Al5_Al5[-1]*w[4]) / (Q3_Al5[-1]*w[0]+Q4_Al5[-1]*w[1]+Q2_Al5[-1]*w[2]+Q1_Al5[-1]*w[3]+Al5_Al5[-1]*w[4])

            # Fraktioner
            if Q4A_Al5[-1]+Q4N_Al5[-1] > 0:
                rQ4A = Q4A_Al5[-1]/(Q4A_Al5[-1]+Q4N_Al5[-1])
                rQ4N = Q4N_Al5[-1]/(Q4A_Al5[-1]+Q4N_Al5[-1])
            else:
                rQ4A = 0
                rQ4N = 0

            if Q2A_Al5[-1]+Q2N_Al5[-1] > 0:
                rQ2A = Q2A_Al5[-1]/(Q2A_Al5[-1]+Q2N_Al5[-1])
                rQ2N = Q2N_Al5[-1]/(Q2A_Al5[-1]+Q2N_Al5[-1])
            else:
                rQ2A = 0
                rQ2N = 0

            if Q1AA_Al5[-1]+Q1AN_Al5[-1]+Q1NN_Al5[-1] > 0:
                rQ1AA = Q1AA_Al5[-1]/(Q1AA_Al5[-1]+Q1AN_Al5[-1]+Q1NN_Al5[-1])
                rQ1AN = Q1AN_Al5[-1]/(Q1AA_Al5[-1]+Q1AN_Al5[-1]+Q1NN_Al5[-1])
                rQ1NN = Q1NN_Al5[-1]/(Q1AA_Al5[-1]+Q1AN_Al5[-1]+Q1NN_Al5[-1])
            else:
                rQ1AA = 0
                rQ1AN = 0
                rQ1NN = 0

            # Draws
            if (Q4_Al5[-1] + Q2_Al5[-1] + 2*Q1_Al5[-1] + 3*Q0_Al5[-1]) < Q3[0]*w[5]:
                P = 1
            else:
                P = 0

            if Q3_Al5[-1] - pQ3 - pQ4*pQ3 > 0:
                next_Q3_c = Q3_Al5[-1] - pQ3 - pQ4*pQ3
                next_Q3_mod = Q3_Al5[-1] - pQ3 - pQ4*pQ3

            else:
                next_Q3_c = 0
                next_Q3_mod = 0

            if Q4_Al5[-1] + pQ3*P - pQ4 + pQ4*pQ3*P - pQ4**2 > 0:
                next_Q4_c = Q4_Al5[-1] + pQ3*P - pQ4 + pQ4*pQ3*P - pQ4**2
                next_Q4N_mod = Q4N_Al5[-1] + pQ3*P - pQ4*rQ4N + (pQ3*P - pQ4*rQ4N)*pQ4
                next_Q4A_mod = Q4A_Al5[-1] - pQ4*rQ4A - (pQ4*rQ4A)*pQ4

            else:
                next_Q4_c = 0
                next_Q4N_mod = 0
                next_Q4A_mod = 0


            if Q2_Al5[-1] + pQ4 - pQ2 + pQ3*(1-P) - pQ4*pQ2 + pQ4*pQ3*(1-P) + pQ4**2 > 0:
                next_Q2_c = Q2_Al5[-1] + pQ4 - pQ2 + pQ3*(1-P) - pQ4*pQ2 + pQ4*pQ3*(1-P) + pQ4**2
                next_Q2N_mod = Q2N_Al5[-1] + pQ4*rQ4N + pQ3*(1-P) - pQ2*rQ2N + (pQ4*rQ4N + pQ3*(1-P) - pQ2*rQ2N)*pQ4
                next_Q2A_mod = Q2A_Al5[-1] - pQ2*rQ2A + (- pQ2*rQ2A)*pQ4

            else:
                next_Q2_c = 0
                next_Q2N_mod = 0
                next_Q2A_mod = 0


            if Q1_Al5[-1] + pQ2 - pQ1 + pQ4*pQ2 - pQ4*pQ1 < 0:
                next_Q1_c = 0
                next_Q1NN_mod = 0
                next_Q1AN_mod = 0
                next_Q1AA_mod = 0

            else:
                next_Q1_c = Q1_Al5[-1] + pQ2 - pQ1 + pQ4*pQ2 - pQ4*pQ1
                next_Q1NN_mod = Q1NN_Al5[-1] + pQ2*rQ2N - pQ1*rQ1NN + (pQ2*rQ2N - pQ1*rQ1NN)*pQ4
                next_Q1AN_mod = Q1AN_Al5[-1] + pQ2*rQ2A - pQ1*rQ1AN + (pQ2*rQ2A - pQ1*rQ1AN)*pQ4
                next_Q1AA_mod = Q1AA_Al5[-1] - pQ1*rQ1AA - pQ1*rQ1AA * pQ4

            if Q0_Al5[-1] + pQ1 + pQ4*pQ1 < 0:
                next_Q0_c = 0
                next_Q0NNN_mod = 0
                next_Q0ANN_mod = 0
                next_Q0AAN_mod = 0
                next_Q0AAA_mod = 0

            else:
                next_Q0_c = Q0_Al5[-1] + pQ1 + pQ4*pQ1
                next_Q0NNN_mod = Q0NNN_Al5[-1] + pQ1*rQ1NN + pQ1*rQ1NN*pQ4
                next_Q0ANN_mod = Q0ANN_Al5[-1] + pQ1*rQ1AN + pQ1*rQ1AN*pQ4
                next_Q0AAN_mod = Q0AAN_Al5[-1] + pQ1*rQ1AA + pQ1*rQ1AA*pQ4
                next_Q0AAA_mod = Q0AAA_Al5[-1]


            if Al5_Al5[-1] - 0.25*pAl5 - pQ4*0.25*pAl5 > 0:
                next_Al5_c = Al5_Al5[-1] - 0.25*pAl5 - pQ4*0.25*pAl5
                next_Al5 = Al5[-1] - 0.25*pAl5 - 0.25*pQ4*pAl5
            else:
                next_Al5_c = 0
                next_Al5 = Al5[-1]

            if Al4_Al5[-1] + 0.25*pAl5 + pQ4*0.25*pAl5 > 0:
                next_Al4_c = Al4_Al5[-1] + 0.25*pAl5 + pQ4*0.25*pAl5
                next_Al4 = Al4[-1] + 0.25*pAl5 + pQ4*0.25*pAl5
            else:
                next_Al4_c = 0
                next_Al4 = Al4[-1]

            next_Q3 = next_Q3_mod + next_Q4A_mod + next_Q2A_mod + next_Q1AA_mod + next_Q0AAA_mod
            next_Q4 = next_Q4N_mod
            next_Q2 = next_Q2N_mod + next_Q1AN_mod + next_Q0AAN_mod
            next_Q1 = next_Q1NN_mod + next_Q0ANN_mod
            next_Q0 = next_Q0NNN_mod

            Q3_c.append(next_Q3_c)
            Q4_c.append(next_Q4_c)
            Q2_c.append(next_Q2_c)
            Q1_c.append(next_Q1_c)
            Q0_c.append(next_Q0_c)
            Al5_c.append(next_Al5_c)
            Al4_c.append(next_Al4_c)

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
        Q3_m = Q3_c[ind]
        Q4_m = Q4_c[ind]
        Q2_m = Q2_c[ind]
        Q1_m = Q1_c[ind]
        Q0_m = Q0_c[ind]
        Al5_m = Al5_c[ind]
        Al4_m = Al4_c[ind]


        SSSE = (((Q3_m - Q3_data[R_ind])**2)+((Q4_m - Q4_data[R_ind])**2)+((Q2_m - Q2_data[R_ind])**2)+((Q1_m - Q1_data[R_ind])**2)+((Q0_m - Q0_data[R_ind])**2)+((Al5_m - Al5_data[R_ind])**2)+((Al4_m - Al4_data[R_ind])**2))
        SSE.append(SSSE)

    return sum(SSE)

SSElist =[]
it = 1
def print_fun(x, f, accepted):
    global it
    global SSElist
    print("SSE = {} for {}".format(f, it))
    it += 1
    SSElist.append(f)

#w1 = [0.5, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.5]

#res = scipy.optimize.basinhopping(model, w1, niter=250, T=1.0, stepsize=0.5, minimizer_kwargs=None, take_step=None, accept_test=None, callback=print_fun, interval=50, disp=False, niter_success=None)

#w = [1, abs(res.x[0]), abs(res.x[1]), abs(res.x[2]), abs(res.x[3]), abs(res.x[4]),  abs(res.x[5]), abs(res.x[6]),  abs(res.x[7]),  abs(res.x[8]),  abs(res.x[9])]

w = [1,0.00107329537489,9.30099817836e-05,1.45510381372,10.6235393896,0.46908402526,1,1.276455,(0.000108473378891/151.2),0.640873,68.40608]

RMSElist = []

for i in SSElist:
    RMSE = math.sqrt(i/(7*len(R_data)))
    RMSElist.append(RMSE)

tab = tt.Texttable()
headings = ['W','Al5','Na']
tab.header(headings)
names = ['Q3', 'Q4', 'Q2', 'Q1', 'Al5/6' ]
Al5 = [w[6], w[7], w[8], w[9], w[10]]
Na = [w[0], w[1], w[2], w[3], w[4]]
for row in zip(names,Al5,Na):
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

    Q3 = [(100/(i+1)), ]
    Q4 = [0, ]
    Q2 = [0, ]
    Q1 = [0, ]
    Q0 = [0, ]
    Al5 = [100*i/(i+1), ]
    Al4 = [0, ]

    R_ind = R_data.index(i)

    Q3_c = [ ]
    Q4_c = [ ]
    Q2_c = [ ]
    Q1_c = [ ]
    Q0_c = [ ]
    Al5_c = [ ]
    Al4_c = [ ]

#For loop for udregning af g og Q_Al5
    for i in draw_ar:

        Q3_Al5 = [Q3[-1], ]
        Q4_Al5 = [Q4[-1],  ]
        Q2_Al5 = [Q2[-1],  ]
        Q1_Al5 = [Q1[-1],  ]
        Q0_Al5 = [Q0[-1],  ]
        Al5_Al5 = [Al5[-1],  ]
        Al4_Al5 = [Al4[-1],  ]

        #Q4
        Q4A_Al5 = [0, ]
        Q4N_Al5 = [Q4[-1], ]

        #Q2
        Q2A_Al5 = [0, ]
        Q2N_Al5 = [Q2[-1], ]

        #Q1
        Q1NN_Al5 = [Q1[-1], ]
        Q1AN_Al5 = [0, ]
        Q1AA_Al5 = [0, ]

        #Q0
        Q0AAA_Al5 = [0, ]
        Q0AAN_Al5 = [0, ]
        Q0ANN_Al5 = [0, ]
        Q0NNN_Al5 = [Q0[-1], ]

        #Al draw
        Al_draw = [0, ]

        while (Al_draw[-1]/3)*100/(100+(Al_draw[-1]/3)) < Al5_Al5[-1]*100/(100-Al4_Al5[-1]):
            gQ3 = (Q3_Al5[-1]*w[6]) / (Q3_Al5[-1]*w[6]+Q4_Al5[-1]*w[7]+Q2_Al5[-1]*w[8]+Q1_Al5[-1]*w[9]+Al5_Al5[-1]*w[10])
            gQ4 = (Q4_Al5[-1]*w[7]) / (Q3_Al5[-1]*w[6]+Q4_Al5[-1]*w[7]+Q2_Al5[-1]*w[8]+Q1_Al5[-1]*w[9]+Al5_Al5[-1]*w[10])
            gQ2 = (Q2_Al5[-1]*w[8]) / (Q3_Al5[-1]*w[6]+Q4_Al5[-1]*w[7]+Q2_Al5[-1]*w[8]+Q1_Al5[-1]*w[9]+Al5_Al5[-1]*w[10])
            gQ1 = (Q1_Al5[-1]*w[9]) / (Q3_Al5[-1]*w[6]+Q4_Al5[-1]*w[7]+Q2_Al5[-1]*w[8]+Q1_Al5[-1]*w[9]+Al5_Al5[-1]*w[10])
            gAl5 = (Al5_Al5[-1]*w[10]) / (Q3_Al5[-1]*w[6]+Q4_Al5[-1]*w[7]+Q2_Al5[-1]*w[8]+Q1_Al5[-1]*w[9]+Al5_Al5[-1]*w[10])


            if Q4A_Al5[-1]+Q4N_Al5[-1] > 0:
                rQ4A = Q4A_Al5[-1]/(Q4A_Al5[-1]+Q4N_Al5[-1])
                rQ4N = Q4N_Al5[-1]/(Q4A_Al5[-1]+Q4N_Al5[-1])
            else:
                rQ4A = 0
                rQ4N = 0

            if Q2A_Al5[-1]+Q2N_Al5[-1] > 0:
                rQ2A = Q2A_Al5[-1]/(Q2A_Al5[-1]+Q2N_Al5[-1])
                rQ2N = Q2N_Al5[-1]/(Q2A_Al5[-1]+Q2N_Al5[-1])
            else:
                rQ2A = 0
                rQ2N = 0

            if Q1AA_Al5[-1]+Q1AN_Al5[-1]+Q1NN_Al5[-1] > 0:
                rQ1AA = Q1AA_Al5[-1]/(Q1AA_Al5[-1]+Q1AN_Al5[-1]+Q1NN_Al5[-1])
                rQ1AN = Q1AN_Al5[-1]/(Q1AA_Al5[-1]+Q1AN_Al5[-1]+Q1NN_Al5[-1])
                rQ1NN = Q1NN_Al5[-1]/(Q1AA_Al5[-1]+Q1AN_Al5[-1]+Q1NN_Al5[-1])
            else:
                rQ1AA = 0
                rQ1AN = 0
                rQ1NN = 0

            # overgang
            if (Q4_Al5[-1] + Q2_Al5[-1] + 2*Q1_Al5[-1] + 3*Q0_Al5[-1]) < Q3[0]*w[5]:
                P = 1
            else:
                P = 0

            # Draws

            if Q3_Al5[-1] - gQ3 + (- gQ3)*gQ4 > 0:
                next_Q3_Al5 = Q3_Al5[-1] - gQ3 + (- gQ3)*gQ4

            else:
                next_Q3_Al5 = 0

            if Q4_Al5[-1] + gQ3*P - gQ4 + (gQ3*P - gQ4)*gQ4 > 0:
                next_Q4A_Al5 = Q4A_Al5[-1] + gQ3*P - gQ4*rQ4A + (gQ3*P - gQ4*rQ4A)*gQ4
                next_Q4N_Al5 = Q4N_Al5[-1] - gQ4*rQ4N - (gQ4*rQ4N)*gQ4
                next_Q4_Al5 = next_Q4A_Al5 + next_Q4N_Al5
            else:
                next_Q4A_Al5 = 0
                next_Q4N_Al5 = 0
                next_Q4_Al5 = 0

            if Q2_Al5[-1] + gQ4 + gQ3*(1-P) - gQ2 + (gQ4 + gQ3*(1-P) - gQ2)*gQ4 > 0:
                next_Q2A_Al5 = Q2A_Al5[-1] + gQ4*rQ4A + gQ3*(1-P) - gQ2*rQ2A + (gQ4*rQ4A + gQ3*(1-P) - gQ2*rQ2A)*gQ4
                next_Q2N_Al5 = Q2N_Al5[-1] + gQ4*rQ4N - gQ2*rQ2N + (gQ4*rQ4N - gQ2*rQ2N)*gQ4
                next_Q2_Al5 = next_Q2A_Al5 + next_Q2N_Al5
            else:
                next_Q2A_Al5 = 0
                next_Q2N_Al5 = 0
                next_Q2_Al5 = 0

            if Q1_Al5[-1] + gQ2 - gQ1 + (gQ2 - gQ1)*gQ4 < 0:
                next_Q1AA_Al5 = 0
                next_Q1AN_Al5 = 0
                next_Q1NN_Al5 = 0
                next_Q1_Al5 = 0
            else:
                next_Q1AA_Al5 = Q1AA_Al5[-1] + gQ2*rQ2A - gQ1*rQ1AA + (gQ2*rQ2A - gQ1*rQ1AA)*gQ4
                next_Q1AN_Al5 = Q1AN_Al5[-1] + gQ2*rQ2N - gQ1*rQ1AN + (gQ2*rQ2N - gQ1*rQ1AN)*gQ4
                next_Q1NN_Al5 = Q1NN_Al5[-1] - gQ1*rQ1NN - (gQ1*rQ1NN)*gQ4
                next_Q1_Al5 = next_Q1AA_Al5 + next_Q1AN_Al5 + next_Q1NN_Al5

            if Q0_Al5[-1] + gQ1 + (gQ1)*gQ4 < 0:
                next_Q0AAA_Al5 = 0
                next_Q0AAN_Al5 = 0
                next_Q0ANN_Al5 = 0
                next_Q1_Al5 = 0
            else:
                next_Q0AAA_Al5 = Q0AAA_Al5[-1] + gQ1*rQ1AA + (gQ1*rQ1AA)*gQ4
                next_Q0AAN_Al5 = Q0AAN_Al5[-1] + gQ1*rQ1AN + (gQ1*rQ1AN)*gQ4
                next_Q0ANN_Al5 = Q0ANN_Al5[-1] + gQ1*rQ1NN + (gQ1*rQ1NN)*gQ4
                next_Q0_Al5 = next_Q0AAA_Al5 + next_Q0AAN_Al5 + next_Q0ANN_Al5 + Q0NNN_Al5[-1]

            if Al5_Al5[-1] - gAl5 + (-gAl5)*gQ4 > 0:
                next_Al5_Al5 = Al5_Al5[-1] - gAl5 + (-gAl5)*gQ4
            else:
                next_Al5_Al5 = 0

            if Al4_Al5[-1] + gAl5 + (gAl5)*gQ4 > 0:
                next_Al4_Al5 = Al4_Al5[-1] + gAl5 + (gAl5)*gQ4
            else:
                next_Al4_Al5 = 0

            Q3_Al5.append(next_Q3_Al5)

            Q4_Al5.append(next_Q4_Al5)
            Q4A_Al5.append(next_Q4A_Al5)
            Q4N_Al5.append(next_Q4N_Al5)

            Q2_Al5.append(next_Q2_Al5)
            Q2A_Al5.append(next_Q2A_Al5)
            Q2N_Al5.append(next_Q2N_Al5)

            Q1_Al5.append(next_Q1_Al5)
            Q1AA_Al5.append(next_Q1AA_Al5)
            Q1AN_Al5.append(next_Q1AN_Al5)
            Q1NN_Al5.append(next_Q1NN_Al5)

            Q0_Al5.append(next_Q0_Al5)
            Q0AAA_Al5.append(next_Q0AAA_Al5)
            Q0AAN_Al5.append(next_Q0AAN_Al5)
            Q0ANN_Al5.append(next_Q0ANN_Al5)

            Al5_Al5.append(next_Al5_Al5)
            Al4_Al5.append(next_Al4_Al5)

            next_Al_draw = Al_draw[-1]+1
            Al_draw.append(next_Al_draw)

        ## Residual Al draw ##
        if Al4_Al5[-1] + Al5_Al5[-1] > 0:
            aa = ((-300*(Al5_Al5[-1]*100/(100-Al4_Al5[-1])))/(-100+(Al5_Al5[-1]*100/(100-Al4_Al5[-1]))))-(-300*(Al5_Al5[-2]*100/(100-Al4_Al5[-2])))/(-100+(Al5_Al5[-2]*100/(100-Al4_Al5[-2])))
            b = -aa*Al_draw[-1]+(-300*(Al5_Al5[-1]*100/(100-Al4_Al5[-1])))/(-100+(Al5_Al5[-1]*100/(100-Al4_Al5[-1])))
            balance_draw = (-b/(aa-1))-Al_draw[-2]
        else:
            balance_draw = 0
    #        residual_draw = (-300*(Al5_Al5[-2]*100/(100-Al4_Al5[-2])))/(-100+(Al5_Al5[-2]*100/(100-Al4_Al5[-2])))
    #        gAl5 = (Al5_Al5[-2]*w[10]) / (Q3_Al5[-2]*w[6]+Q4_Al5[-2]*w[7]+Q2_Al5[-2]*w[8]+Q1_Al5[-2]*w[9]+Al5_Al5[-2]*w[10])
    #        gQ4 = (Q4_Al5[-2]*w[7]) / (Q3_Al5[-2]*w[6]+Q4_Al5[-2]*w[7]+Q2_Al5[-2]*w[8]+Q1_Al5[-2]*w[9]+Al5_Al5[-2]*w[10])
    #        balance_draw = (residual_draw - (Al_draw[-2]))/(gAl5 + gAl5*gQ4 + 1)
    #        testlist.append(residual_draw - (Al_draw[-2]))

        gQ3 = balance_draw*(Q3_Al5[-1]*w[6]) / (Q3_Al5[-1]*w[6]+Q4_Al5[-1]*w[7]+Q2_Al5[-1]*w[8]+Q1_Al5[-1]*w[9]+Al5_Al5[-1]*w[10])
        gQ4 = balance_draw*(Q4_Al5[-1]*w[7]) / (Q3_Al5[-1]*w[6]+Q4_Al5[-1]*w[7]+Q2_Al5[-1]*w[8]+Q1_Al5[-1]*w[9]+Al5_Al5[-1]*w[10])
        gQ2 = balance_draw*(Q2_Al5[-1]*w[8]) / (Q3_Al5[-1]*w[6]+Q4_Al5[-1]*w[7]+Q2_Al5[-1]*w[8]+Q1_Al5[-1]*w[9]+Al5_Al5[-1]*w[10])
        gQ1 = balance_draw*(Q1_Al5[-1]*w[9]) / (Q3_Al5[-1]*w[6]+Q4_Al5[-1]*w[7]+Q2_Al5[-1]*w[8]+Q1_Al5[-1]*w[9]+Al5_Al5[-1]*w[10])
        gAl5 = balance_draw*(Al5_Al5[-1]*w[10]) / (Q3_Al5[-1]*w[6]+Q4_Al5[-1]*w[7]+Q2_Al5[-1]*w[8]+Q1_Al5[-1]*w[9]+Al5_Al5[-1]*w[10])

        if Q4A_Al5[-1]+Q4N_Al5[-1] > 0:
            rQ4A = Q4A_Al5[-1]/(Q4A_Al5[-1]+Q4N_Al5[-1])
            rQ4N = Q4N_Al5[-1]/(Q4A_Al5[-1]+Q4N_Al5[-1])
        else:
            rQ4A = 0
            rQ4N = 0

        if Q2A_Al5[-1]+Q2N_Al5[-1] > 0:
            rQ2A = Q2A_Al5[-1]/(Q2A_Al5[-1]+Q2N_Al5[-1])
            rQ2N = Q2N_Al5[-1]/(Q2A_Al5[-1]+Q2N_Al5[-1])
        else:
            rQ2A = 0
            rQ2N = 0

        if Q1AA_Al5[-1]+Q1AN_Al5[-1]+Q1NN_Al5[-1] > 0:
            rQ1AA = Q1AA_Al5[-1]/(Q1AA_Al5[-1]+Q1AN_Al5[-1]+Q1NN_Al5[-1])
            rQ1AN = Q1AN_Al5[-1]/(Q1AA_Al5[-1]+Q1AN_Al5[-1]+Q1NN_Al5[-1])
            rQ1NN = Q1NN_Al5[-1]/(Q1AA_Al5[-1]+Q1AN_Al5[-1]+Q1NN_Al5[-1])
        else:
            rQ1AA = 0
            rQ1AN = 0
            rQ1NN = 0

        #overgang
        if (Q4_Al5[-1] + Q2_Al5[-1] + 2*Q1_Al5[-1] + 3*Q0_Al5[-1]) < Q3[0]*w[5]:
            P = 1
        else:
            P = 0

           # Draws

        if Q3_Al5[-1] - gQ3 + (- gQ3)*gQ4 > 0:
            next_Q3_Al5 = Q3_Al5[-1] - gQ3 + (- gQ3)*gQ4

        else:
            next_Q3_Al5 = 0

        if Q4_Al5[-1] + gQ3*P - gQ4 + (gQ3*P - gQ4)*gQ4 > 0:
            next_Q4A_Al5 = Q4A_Al5[-1] + gQ3*P - gQ4*rQ4A + (gQ3*P - gQ4*rQ4A)*gQ4
            next_Q4N_Al5 = Q4N_Al5[-1] - gQ4*rQ4N - (gQ4*rQ4N)*gQ4
            next_Q4_Al5 = next_Q4A_Al5 + next_Q4N_Al5
        else:
            next_Q4A_Al5 = 0
            next_Q4N_Al5 = 0
            next_Q4_Al5 = 0

        if Q2_Al5[-1] + gQ4 + gQ3*(1-P) - gQ2 + (gQ4 + gQ3*(1-P) - gQ2)*gQ4 > 0:
            next_Q2A_Al5 = Q2A_Al5[-1] + gQ4*rQ4A + gQ3*(1-P) - gQ2*rQ2A + (gQ4*rQ4A + gQ3*(1-P) - gQ2*rQ2A)*gQ4
            next_Q2N_Al5 = Q2N_Al5[-1] + gQ4*rQ4N - gQ2*rQ2N + (gQ4*rQ4N - gQ2*rQ2N)*gQ4
            next_Q2_Al5 = next_Q2A_Al5 + next_Q2N_Al5
        else:
            next_Q2A_Al5 = 0
            next_Q2N_Al5 = 0
            next_Q2_Al5 = 0

        if Q1_Al5[-1] + gQ2 - gQ1 + (gQ2 - gQ1)*gQ4 < 0:
            next_Q1AA_Al5 = 0
            next_Q1AN_Al5 = 0
            next_Q1NN_Al5 = 0
            next_Q1_Al5 = 0
        else:
            next_Q1AA_Al5 = Q1AA_Al5[-1] + gQ2*rQ2A - gQ1*rQ1AA + (gQ2*rQ2A - gQ1*rQ1AA)*gQ4
            next_Q1AN_Al5 = Q1AN_Al5[-1] + gQ2*rQ2N - gQ1*rQ1AN + (gQ2*rQ2N - gQ1*rQ1AN)*gQ4
            next_Q1NN_Al5 = Q1NN_Al5[-1] - gQ1*rQ1NN - (gQ1*rQ1NN)*gQ4
            next_Q1_Al5 = next_Q1AA_Al5 + next_Q1AN_Al5 + next_Q1NN_Al5

        if Q0_Al5[-1] + gQ1 + (gQ1)*gQ4 < 0:
            next_Q0AAA_Al5 = 0
            next_Q0AAN_Al5 = 0
            next_Q0ANN_Al5 = 0
            next_Q1_Al5 = 0
        else:
            next_Q0AAA_Al5 = Q0AAA_Al5[-1] + gQ1*rQ1AA + (gQ1*rQ1AA)*gQ4
            next_Q0AAN_Al5 = Q0AAN_Al5[-1] + gQ1*rQ1AN + (gQ1*rQ1AN)*gQ4
            next_Q0ANN_Al5 = Q0ANN_Al5[-1] + gQ1*rQ1NN + (gQ1*rQ1NN)*gQ4
            next_Q0_Al5 = next_Q0AAA_Al5 + next_Q0AAN_Al5 + next_Q0ANN_Al5 + Q0NNN_Al5[-1]

        if Al5_Al5[-1] - gAl5 + (-gAl5)*gQ4 > 0:
            next_Al5_Al5 = Al5_Al5[-1] - gAl5 + (-gAl5)*gQ4
        else:
            next_Al5_Al5 = 0

        if Al4_Al5[-1] + gAl5 + (gAl5)*gQ4 > 0:
            next_Al4_Al5 = Al4_Al5[-1] + gAl5 + (gAl5)*gQ4
        else:
            next_Al4_Al5 = 0

        Q3_Al5.append(next_Q3_Al5)

        Q4_Al5.append(next_Q4_Al5)
        Q4A_Al5.append(next_Q4A_Al5)
        Q4N_Al5.append(next_Q4N_Al5)

        Q2_Al5.append(next_Q2_Al5)
        Q2A_Al5.append(next_Q2A_Al5)
        Q2N_Al5.append(next_Q2N_Al5)

        Q1_Al5.append(next_Q1_Al5)
        Q1AA_Al5.append(next_Q1AA_Al5)
        Q1AN_Al5.append(next_Q1AN_Al5)
        Q1NN_Al5.append(next_Q1NN_Al5)

        Q0_Al5.append(next_Q0_Al5)
        Q0AAA_Al5.append(next_Q0AAA_Al5)
        Q0AAN_Al5.append(next_Q0AAN_Al5)
        Q0ANN_Al5.append(next_Q0ANN_Al5)

        Al5_Al5.append(next_Al5_Al5)
        Al4_Al5.append(next_Al4_Al5)

        pQ3 = (Q3_Al5[-1]*w[0]) / (Q3_Al5[-1]*w[0]+Q4_Al5[-1]*w[1]+Q2_Al5[-1]*w[2]+Q1_Al5[-1]*w[3]+Al5_Al5[-1]*w[4])
        pQ4 = (Q4_Al5[-1]*w[1]) / (Q3_Al5[-1]*w[0]+Q4_Al5[-1]*w[1]+Q2_Al5[-1]*w[2]+Q1_Al5[-1]*w[3]+Al5_Al5[-1]*w[4])
        pQ2 = (Q2_Al5[-1]*w[2]) / (Q3_Al5[-1]*w[0]+Q4_Al5[-1]*w[1]+Q2_Al5[-1]*w[2]+Q1_Al5[-1]*w[3]+Al5_Al5[-1]*w[4])
        pQ1 = (Q1_Al5[-1]*w[3]) / (Q3_Al5[-1]*w[0]+Q4_Al5[-1]*w[1]+Q2_Al5[-1]*w[2]+Q1_Al5[-1]*w[3]+Al5_Al5[-1]*w[4])
        pAl5 = (Al5_Al5[-1]*w[4]) / (Q3_Al5[-1]*w[0]+Q4_Al5[-1]*w[1]+Q2_Al5[-1]*w[2]+Q1_Al5[-1]*w[3]+Al5_Al5[-1]*w[4])

        # Fraktioner
        if Q4A_Al5[-1]+Q4N_Al5[-1] > 0:
            rQ4A = Q4A_Al5[-1]/(Q4A_Al5[-1]+Q4N_Al5[-1])
            rQ4N = Q4N_Al5[-1]/(Q4A_Al5[-1]+Q4N_Al5[-1])
        else:
            rQ4A = 0
            rQ4N = 0

        if Q2A_Al5[-1]+Q2N_Al5[-1] > 0:
            rQ2A = Q2A_Al5[-1]/(Q2A_Al5[-1]+Q2N_Al5[-1])
            rQ2N = Q2N_Al5[-1]/(Q2A_Al5[-1]+Q2N_Al5[-1])
        else:
            rQ2A = 0
            rQ2N = 0

        if Q1AA_Al5[-1]+Q1AN_Al5[-1]+Q1NN_Al5[-1] > 0:
            rQ1AA = Q1AA_Al5[-1]/(Q1AA_Al5[-1]+Q1AN_Al5[-1]+Q1NN_Al5[-1])
            rQ1AN = Q1AN_Al5[-1]/(Q1AA_Al5[-1]+Q1AN_Al5[-1]+Q1NN_Al5[-1])
            rQ1NN = Q1NN_Al5[-1]/(Q1AA_Al5[-1]+Q1AN_Al5[-1]+Q1NN_Al5[-1])
        else:
            rQ1AA = 0
            rQ1AN = 0
            rQ1NN = 0

        # Draws
        if (Q4_Al5[-1] + Q2_Al5[-1] + 2*Q1_Al5[-1] + 3*Q0_Al5[-1]) < Q3[0]*w[5]:
            P = 1
        else:
            P = 0

        if Q3_Al5[-1] - pQ3 - pQ4*pQ3 > 0:
            next_Q3_c = Q3_Al5[-1] - pQ3 - pQ4*pQ3
            next_Q3_mod = Q3_Al5[-1] - pQ3 - pQ4*pQ3

        else:
            next_Q3_c = 0
            next_Q3_mod = 0

        if Q4_Al5[-1] + pQ3*P - pQ4 + pQ4*pQ3*P - pQ4**2 > 0:
            next_Q4_c = Q4_Al5[-1] + pQ3*P - pQ4 + pQ4*pQ3*P - pQ4**2
            next_Q4N_mod = Q4N_Al5[-1] + pQ3*P - pQ4*rQ4N + (pQ3*P - pQ4*rQ4N)*pQ4
            next_Q4A_mod = Q4A_Al5[-1] - pQ4*rQ4A - (pQ4*rQ4A)*pQ4

        else:
            next_Q4_c = 0
            next_Q4N_mod = 0
            next_Q4A_mod = 0


        if Q2_Al5[-1] + pQ4 - pQ2 + pQ3*(1-P) - pQ4*pQ2 + pQ4*pQ3*(1-P) + pQ4**2 > 0:
            next_Q2_c = Q2_Al5[-1] + pQ4 - pQ2 + pQ3*(1-P) - pQ4*pQ2 + pQ4*pQ3*(1-P) + pQ4**2
            next_Q2N_mod = Q2N_Al5[-1] + pQ4*rQ4N + pQ3*(1-P) - pQ2*rQ2N + (pQ4*rQ4N + pQ3*(1-P) - pQ2*rQ2N)*pQ4
            next_Q2A_mod = Q2A_Al5[-1] - pQ2*rQ2A + (- pQ2*rQ2A)*pQ4

        else:
            next_Q2_c = 0
            next_Q2N_mod = 0
            next_Q2A_mod = 0


        if Q1_Al5[-1] + pQ2 - pQ1 + pQ4*pQ2 - pQ4*pQ1 < 0:
            next_Q1_c = 0
            next_Q1NN_mod = 0
            next_Q1AN_mod = 0
            next_Q1AA_mod = 0

        else:
            next_Q1_c = Q1_Al5[-1] + pQ2 - pQ1 + pQ4*pQ2 - pQ4*pQ1
            next_Q1NN_mod = Q1NN_Al5[-1] + pQ2*rQ2N - pQ1*rQ1NN + (pQ2*rQ2N - pQ1*rQ1NN)*pQ4
            next_Q1AN_mod = Q1AN_Al5[-1] + pQ2*rQ2A - pQ1*rQ1AN + (pQ2*rQ2A - pQ1*rQ1AN)*pQ4
            next_Q1AA_mod = Q1AA_Al5[-1] - pQ1*rQ1AA - pQ1*rQ1AA * pQ4

        if Q0_Al5[-1] + pQ1 + pQ4*pQ1 < 0:
            next_Q0_c = 0
            next_Q0NNN_mod = 0
            next_Q0ANN_mod = 0
            next_Q0AAN_mod = 0
            next_Q0AAA_mod = 0

        else:
            next_Q0_c = Q0_Al5[-1] + pQ1 + pQ4*pQ1
            next_Q0NNN_mod = Q0NNN_Al5[-1] + pQ1*rQ1NN + pQ1*rQ1NN*pQ4
            next_Q0ANN_mod = Q0ANN_Al5[-1] + pQ1*rQ1AN + pQ1*rQ1AN*pQ4
            next_Q0AAN_mod = Q0AAN_Al5[-1] + pQ1*rQ1AA + pQ1*rQ1AA*pQ4
            next_Q0AAA_mod = Q0AAA_Al5[-1]


        if Al5_Al5[-1] - 0.25*pAl5 - pQ4*0.25*pAl5 > 0:
            next_Al5_c = Al5_Al5[-1] - 0.25*pAl5 - pQ4*0.25*pAl5
            next_Al5 = Al5[-1] - 0.25*pAl5 - 0.25*pQ4*pAl5
        else:
            next_Al5_c = 0
            next_Al5 = Al5[-1]

        if Al4_Al5[-1] + 0.25*pAl5 + pQ4*0.25*pAl5 > 0:
            next_Al4_c = Al4_Al5[-1] + 0.25*pAl5 + pQ4*0.25*pAl5
            next_Al4 = Al4[-1] + 0.25*pAl5 + pQ4*0.25*pAl5
        else:
            next_Al4_c = 0
            next_Al4 = Al4[-1]

        next_Q3 = next_Q3_mod + next_Q4A_mod + next_Q2A_mod + next_Q1AA_mod + next_Q0AAA_mod
        next_Q4 = next_Q4N_mod
        next_Q2 = next_Q2N_mod + next_Q1AN_mod + next_Q0AAN_mod
        next_Q1 = next_Q1NN_mod + next_Q0ANN_mod
        next_Q0 = next_Q0NNN_mod

        Q3_c.append(next_Q3_c)
        Q4_c.append(next_Q4_c)
        Q2_c.append(next_Q2_c)
        Q1_c.append(next_Q1_c)
        Q0_c.append(next_Q0_c)
        Al5_c.append(next_Al5_c)
        Al4_c.append(next_Al4_c)

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
    Q3_m.append(Q3_c[ind])
    Q4_m.append(Q4_c[ind])
    Q2_m.append(Q2_c[ind])
    Q1_m.append(Q1_c[ind])
    Q0_m.append(Q0_c[ind])
    Al5_m.append(Al5_c[ind])
    Al4_m.append(Al4_c[ind])

    if R_ind == 3:
        fig1 = plt.figure(figsize=(12, 7))
        plt.subplot(111)
        plt.plot(M2O, Q3_c, 'r', M2O, Q4_c, 'y', M2O, Q2_c, 'g', M2O, Al4_c, 'm', M2O, Al5_c, 'b', M2O, Q1_c, 'c', M2O, Q0_c, 'k', mod_data[R_ind], Q3_data[R_ind], 'rd', mod_data[R_ind], Q4_data[R_ind], 'yd', mod_data[R_ind], Q2_data[R_ind], 'gd', mod_data[R_ind], Al4_data[R_ind], 'md', mod_data[R_ind], Al5_data[R_ind], 'bd')
        plt.xlabel("M2O (mol %)")
        plt.ylabel("Qn/Al distribution (mol %)")
        plt.title('Distribution')
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)
        plt.savefig("{}_modifier_{}_{}_{}.svg".format(file_name, fil, tid, R_ind))


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
