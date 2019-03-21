import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
#import plotly.plotly as py
import time
import numpy as np
from scipy import fftpack
import numpy.fft as fft
from numpy import *


Znet=0
cap = cv2.VideoCapture("Ltest2.mp4")
#cap=cv2.VideoCapture(0)
ret, old_frame = cap.read()
## get first frame to compair
oldframe_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
height = np.size(old_frame, 0)
width = np.size(old_frame, 1)
scalefactor=4
oldframe_gray=cv2.resize(oldframe_gray,(int(width/scalefactor),int(height/scalefactor)))
motion=[0]
bpm=0
fig = plt.figure()
ax = fig.add_subplot(111)
fig.show()
t=[0]
flow=float(0)
n=65536
lastbmp=0
timenow=0
while(1):

	#print(time.process_time())
	ret, curr = cap.read()
	
#	curr=cv2.resize(curr,(int(width/scalefactor),int(height/scalefactor)))

	frame_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
	frame_gray=cv2.resize(frame_gray,(int(width/scalefactor),int(height/scalefactor)))

	#cv2.imshow('frame',old_frame)
	
	displacementy=[0]
	flow=cv2.calcOpticalFlowFarneback(oldframe_gray , frame_gray,flow=flow, pyr_scale=0.4,levels=1, winsize=12,iterations=2,poly_n=8,poly_sigma=1.2,flags=0)
	oldframe_gray = frame_gray

	#print(flow)
	for x in range(int(900/scalefactor), int(1100/scalefactor),1):
			for y in range(int(600/scalefactor),int(900/scalefactor),1):
				
				Znet+= flow[y,x]
				znow=flow[y,x]
				cv2.line(curr,(x*scalefactor,y*scalefactor),( int(x*scalefactor), int(y*scalefactor+10*znow[1]*scalefactor)),(255,0,0),2)
				cv2.circle(curr,(x*scalefactor,y*scalefactor), 1, (0,0,255), -1)
	ynet=Znet[1]
	print(motion)
	if (len(motion)>200):
		motion=motion[1:]
		t=t[1:]
		
		
		spectrum = fft.fft(motion,n=n)
		print(spectrum)
		spectrum=abs(spectrum)
		freq = fft.fftfreq(len(spectrum),d=1/Fs)

		rangepoint1Hz=int(.15/(Fs/n))
		#print(rangepoint1Hz)
		range3hz=int(3/(Fs/n))
		#print(freq[rangepoint1Hz])
		#print(Fs)
		#print(range3hz)
		freq=freq[rangepoint1Hz:range3hz]
		spectrum=spectrum[rangepoint1Hz:range3hz]
		c = (diff(sign(diff(spectrum))) < 0).nonzero()[0] + 1 # local max
		toppeak=max(spectrum[c])
		for x in c:
			if(toppeak==spectrum[x]):
				break
		bpm=abs(freq[x])
		#if(bpm==0):
		#	bpm=currbpm
		#bpm=(bpm+currbpm)/2
		print("%.3f" %bpm)
		s="Freq: %.3f" %bpm
		fig.suptitle(s, fontsize=32, fontweight='bold')

		
		#print("%.3f" %freq[x])
		#plt.plot(freq,spectrum)
		
		#plt.show()
	motion.append(-ynet)
	
	timenow=timenow+1/30
	#t.append(time.process_time())
	t.append(timenow)
	Fs=1/30
	#Fs=1/((t[len(t)-1]-t[0])/(len(t)-1))
	
	
	ax.plot(t, motion, color='b')
	fig.canvas.draw()
	i=t[len(t)-1]
	ax.set_xlim(left=max(0, i-20), right=i+2)
	ax.set_ylim(bottom=min(motion),top=max(motion))
	
	#print(motion)
	cv2.imshow('frame',curr)
	
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

		
cv2.destroyAllWindows()	
cap.release()		
