import serial
from serial.tools import list_ports
import time
import csv

# Function to add a row to the csv for each run
def build_csv(values, csv_name, first=False):
        # if first time
        if first:
            # Create CSV file
            f = open("data.csv","w",newline='')
            # Add column names
            headers = []
            for k in range(0, kmax, 5):
                headers += ["Acc_x_"+ str(k+1), "Acc_y_"+ str(k+1),  "Acc_z_"+ str(k+1), "Gyro_x_" + str(k+1), "Gyro_y_" + str(k+1), "Gyro_z_" + str(k+1)]
            writer.writerow(headers)
        with open(f, "a", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        # add the values
        writer.writerow(values)

# Identify the correct port
ports = list_ports.comports()
for port in ports: print(port)




# Open the serial com
serialCom = serial.Serial('COM7',9600)

# Toggle DTR to reset the Arduino
serialCom.setDTR(False)
time.sleep(1)
serialCom.flushInput()
serialCom.setDTR(True)

# How many data points to record
kmax = 400

writer = csv.writer(f,delimiter=",")

# Loop through and collect data as it is available
for k in range(0, kmax, 5):
    try:
        # Read the line
        s_bytes = serialCom.readline()
        decoded_bytes = s_bytes.decode("utf-8").strip('\r\n')

        # Parse the line            
        values = [float(x) for x in decoded_bytes.split()]
        print(values)

        # Write the row to the CSV
        writer.writerow(values)

    except:
        print("Error encountered, line was not recorded.")

f.close()
