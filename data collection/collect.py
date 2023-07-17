import serial
from serial.tools import list_ports
import time
import csv

# Identify the correct port
ports = list_ports.comports()
for port in ports: print(port)

# Create CSV file
f = open("data.csv","w",newline='')
f.truncate()

# Open the serial com
serialCom = serial.Serial('COM7',9600)

# Toggle DTR to reset the Arduino
serialCom.setDTR(False)
time.sleep(1)
serialCom.flushInput()
serialCom.setDTR(True)

# How many data points to record
kmax = 10

writer = csv.writer(f,delimiter=",")

# Loop through and collect data as it is available
for k in range(kmax):
    try:
        # Read the line
        s_bytes = serialCom.readline()
        decoded_bytes = s_bytes.decode("utf-8").strip('\r\n')
        # print(decoded_bytes)
        # Parse the line
        if k == 0:
            headers = ["xAcc", "yAcc", "zAcc", "xGyro", "yGyro", "zGyro"]
            writer.writerow(headers)
            
        values = [float(x) for x in decoded_bytes.split()]
        print(values)

        # Write to CSV
        writer.writerow(values)

    except:
        print("Error encountered, line was not recorded.")

f.close()
