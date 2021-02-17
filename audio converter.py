import pyaudio
import wave
import sys
import cv2
import numpy as np
import matplotlib.ticker
import matplotlib.pyplot as plt
from windowing import overlapped_window
from wavwrapper import wavfile, monowrapper
import os


class AudioRecorder(object):
        FORMAT=pyaudio.paInt16
        CHANNELS=2
        SAMPLE_RATE=8000
        CHUNK=1024
        DURATION_SECS=1
        MAX_FRAMES = 7
        filename = None
        audio = None
        stream = None
        frames = []
        
        def __init__(self):
                path='\\dir'

                audioFormat='.wav'           
                fileList=[os.path.join(path,f) for f in os.listdir(path) if f.endswith(audioFormat)]
                for AudioFile in fileList:
                    self.filename=AudioFile
                    self.offline_recorder()  
                while True:
                    c = cv2.waitKey(1)
                    if c == 27:
                            print 'finished'
                            cv2.destroyAllWindows()
                            break
        def offline_recorder(self):          
                self.get_spectrogram()
                self.save_spectrogram()
        

        def get_spectrogram(self, window_size=256, scale="log"):
                hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * (np.arange(window_size)) / window_size)
                overlap = 4
                Y = []
                acc = 0
    
                wav = wave.open(self.filename, 'r')
                w = wavfile(wav)

                for x in overlapped_window(monowrapper(w), window_size, overlap):
                        y = np.fft.rfft(x * hann)[:window_size//2]
                        Y.append(y)
                        acc += window_size
    
                Y = np.column_stack(Y)
                Y = np.absolute(Y) * 2.0 / np.sum(hann)
                Y = Y / np.power(2.0, (8 * w.get_param('sampwidth') - 1))
                Y = (20.0 * np.log10(Y)).clip(-120)

                t = np.arange(0, Y.shape[1], dtype=np.float) * window_size / w.get_param('framerate') / overlap    
                f = np.arange(0, window_size / 2, dtype=np.float) * w.get_param('framerate') / window_size
    
                ax = plt.subplot(111)
                plt.pcolormesh(t, f, Y, vmin=-120, vmax=0)
    
                if scale == 'log':
                        yscale = 0.25
                        if matplotlib.__version__[0:3] == '1.3': yscale = 1
                        plt.yscale('symlog', linthreshy=100, linscaley=yscale)
                        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

                plt.axis('off')

        def trim_spectrogram(self, imgray):
                lower_xx = 0
                lower_yy = 0
                higher_xx = 0
                higher_yy = 0
                rows, cols = imgray.shape
        
                for i in range(0,rows):
                        for j in range(0,cols):
                                if imgray[i,j] != 255 and imgray[i,j] != 15: 
                                        lower_xx = i
                                        lower_yy = j
                                        break
                for i in range(0,cols):
                        for j in range(0,rows):
                                if imgray[j,i] != 255 and imgray[j,i] != 15: 
                                        higher_xx = j
                                        higher_yy = i
                                        break

                crop = imgray[higher_xx:lower_xx, lower_yy:higher_yy]
                return crop

        def save_spectrogram(self):
                array = self.filename.split(".wav")
                img_filename = array[0]+".png"
                plt.savefig(img_filename)

                im = cv2.imread(img_filename)
                img_filename=img_filename[:-16]#19
                img_filename=img_filename.replace("SJ_Audio","speaking_person_name")
                img_filename=img_filename+'.png'
                im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                im_trimmed = self.trim_spectrogram(im_gray)
                im_resized = cv2.resize(im_trimmed, (100,70))
                
                cv2.imwrite(img_filename, im_resized)
                print img_filename
                cv2.imshow('Spectrogram', im_trimmed)                
recorder = AudioRecorder()
