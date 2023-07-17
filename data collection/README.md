# Data set

The dataset is built in this way:

>- we have two different signals (accelerometer and gyroscope);
>- 3 axis for each signal (x, y, z);
>- the sampling rate of both sensors are fixed since, in order to simplify the data set and make it easier to manipulate, we collect data every 20 seconds and record 5 timestamps (values of the 2 signals in the 3 different axis) per second;
>- in other words we end-up with 400 columns (features) for each axis of a patricular signal, i.e. a total of $400 \times 3 \times 2$ features;
>- each row will represent a specific observation (fall or not fall).