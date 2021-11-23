# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 14:08:27 2021

@author: Jonah Post
"""




CyclotronFreq = ChargeDensity*MagneticFieldStrength / (Energy+Pressure)
PlasmonFreq = np.sqrt(ChargeDensity**2 / (Energy+Pressure))

TauT = SigmaXY/(CyclotronFreq*SigmaXX)
TauL = SigmaXX * (1 + (SigmaXY/SigmaXX)**2)/ (PlasmonFreq**2)