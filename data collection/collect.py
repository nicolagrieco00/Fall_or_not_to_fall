import serial
from serial.tools import list_ports
import time
import csv
import pandas as pd

# Build CSV
def build_csv(csv_name, kmax=400):
    # Create CSV file
    f = open(csv_name,"w",newline='')
    writer = csv.writer(f, delimiter=",")
    # Add column names
    headers = ["xAcc", "yAcc", "zAcc", "xGyro", "yGyro", "zGyro", "label"]
    writer.writerow(headers)
    f.close()
    return csv_name

# Function to add a row to the csv for each run
def add_row(values, csv_name):
    with open(csv_name, "a", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        # add the values
        writer.writerow(values)


def collect_data(label, flag=False):
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
    kmax = 2000
    data = "data.csv"
    # Loop through and collect data as it is available
    for k in range(0, kmax):
        if k==0 and flag:
            build_csv("data.csv")
        try:
            # Read the line
            s_bytes = serialCom.readline()
            decoded_bytes = s_bytes.decode("utf-8").strip('\r\n')

            # Parse the line            
            values = [float(x) for x in decoded_bytes.split()]
            values.append(label)
            print(values)

            if k%5==0:
                add_row(values, data)

        except:
            print("Error encountered, line was not recorded.")
            if k%5==0:
                add_row(values, data)

    

if __name__=="__main__":
    label = input()
    collect_data(label)