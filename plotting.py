# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 20:47:44 2021

@author: cjove
"""

"""
Plotting del resultado de la identificación de la transición de sedestación a 
bipedestación
"""
peaks = peaks_gyro(acc_filt)
thres_method = sit_to_stand_threshold(acc_filt)
det_sts = sit_to_stand(peaks, thres_method) 
x = range(0,len(thres_method),1)        
for y in x:
    plt.plot(acc_filt['Waist_Gx'][y])
    plt.plot(det_sts[y][0], acc_filt['Waist_Gx'][y][det_sts[y][0]], "o")
    plt.plot(det_sts[y][1], acc_filt['Waist_Gx'][y][det_sts[y][1]], "o")
    plt.show()

"""
Plotting del resultado de la identificación de la transición de sedestación a 
bipedestación
"""

gy_invertidos_d = invertir(acc_filt['Right_Shank_Gy'], index_steps)
gy_invertidos_i = invertir(acc_filt['Left_Shank_Gy'], index_steps)

gait_d = gait_cycles(gy_invertidos_d, index_steps)
gait_i = gait_cycles(gy_invertidos_i, index_steps)

z = np.arange(0,474,1)
for y in z:
    m = np.arange(0, len(index_steps[y]))
    for x in m:
        n = np.arange(0, len(gait_d[y][x]),1)
        if len(n) == 0:
            break
        else:
            for w in n:
                plt.plot(gy_invertidos_d[y][index_steps[y][x][0]:index_steps[y][x][-1]])
                plt.plot(gait_d[y][x][w],gy_invertidos_d[y][index_steps[y][x][0]:index_steps[y][x][-1]][gait_d[y][x][w]], "o")
                #plt.plot(step_derecha[1][y][x],gy_invertidos_d[y][index_steps[y][x][0]:index_steps[y][x][-1]][step_derecha[1][y][x]], "o")
                plt.show()
