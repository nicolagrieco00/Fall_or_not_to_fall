# Data set

The dataset is built in this way:

>- we have two different signals (accelerometer and gyroscope);
>- 3 axis for each signal (x, y, z);
>- the sampling rate of both sensors are fixed since, in order to simplify the data set and make it easier to manipulate, we collect data every 20 seconds, recording 100 values per second (for a total of 2000 timestamps). We then decide to only store and record 400 timestamps (values of the 2 signals in the 3 different axis) for each observation (experiment) with an effective sampling rate of 20 values per second ($\sim$ 20% of the sensors default collection rate);
>- in other words we end-up with a timeseries of length 400  for each axis of a patricular signal, i.e. a total of $400 \times 3 \times 2$ features;
>- each row will represent a specific observation (fall or not fall).

Classes of falls:

>- fall -> grave fall;
>- rfall -> grave all'indietro;
>- lfall -> grave laterale;
>- light -> caduta leggera;
>- sit
>- walk
>- step