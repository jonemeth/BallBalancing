import time
import threading
import serial
import numpy as np

def linReg( x, y ):
   sx = np.sum(x)
   sy = np.sum(y)
   sx2 = np.sum( np.square(x) )
   sxy = np.sum( np.multiply(x,y) )
   N = len(x)

   denom = ( N*sx2 - sx**2 )

   if abs(denom) < 0.0001:
     print('para', flush=True)
     m = 0.0
   else:
     m =( N*sxy - sx*sy ) / denom
   b = (sy - m*sx) / N

   return m, b
   
ser = serial.Serial('/dev/ttyACM0', 9600)

class Sensor:
    def __init__(self):
        self.thread = threading.Thread(target=self.worker)
        self.ser = ser
        self.ser.reset_input_buffer()
        self.thread.start()
        
        self.listX = []
        self.listY = []
        self.listT = []
        self.lock = threading.Lock()

    def worker(self):

        while True:
            try:
                self.ser.reset_input_buffer()
                self.ser.readline()
                r = self.ser.readline().decode("utf-8")
                result=r.split(" , ")
            except:
                continue

            if 2 != len( result ):
                continue

            try:
                X = int(result[0]) / 1000
                Y = int(result[1]) / 1000
            except:
                continue
                
            self.lock.acquire()
            
            try:
                T = time.time( )
                while len(self.listT) > 3 and T-self.listT[0] > 0.2:
                    self.listT.pop(0)
                    self.listX.pop(0)
                    self.listY.pop(0)

                self.listT.append(T)
                self.listX.append(X)
                self.listY.append(Y)
                
            finally:
                self.lock.release()

    def get(self):
        if len(self.listT) <= 0:
            return 0, 0, 0, 0, 0, 0

        self.lock.acquire()

        try:
            T = time.time( )

            relListT = (np.array(self.listT) - self.listT[0]).tolist()
            mx, bx = linReg( relListT, self.listX )
            my, by = linReg( relListT, self.listY )

            #filter
            if len(self.listT) > 3:

                diffsX = [mx*relListT[i]+bx - self.listX[i] for i in range(len(relListT))]
                diffsY = [my*relListT[i]+by - self.listY[i] for i in range(len(relListT))]
                dXstd = np.std(diffsX)
                dYstd = np.std(diffsY)

                fTx = [ relListT[i] for i in range(len(relListT)) if abs(diffsX[i])<2.0*dXstd ]
                fX = [ self.listX[i] for i in range(len(relListT)) if abs(diffsX[i])<2.0*dXstd ]
 
                fTy = [ relListT[i] for i in range(len(relListT)) if abs(diffsY[i])<2.0*dYstd ]
                fY = [ self.listY[i] for i in range(len(relListT)) if abs(diffsY[i])<2.0*dYstd ]

                if len(fX) >= 3:
                    mx, bx = linReg( fTx, fX )
                if len(fY) >= 3:
                    my, by = linReg( fTy, fY )


                relT = T - self.listT[0]

                ballX = mx*relT + bx
                ballY = my*relT + by
                speedX = mx
                speedY = my
            else:
                return 0, 0, 0, 0, 0, 0
        finally:
            self.lock.release()

        return self.listX[-1], self.listY[-1], ballX, ballY, speedX, speedY
