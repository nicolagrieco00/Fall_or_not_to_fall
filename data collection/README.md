# Data set

If you want to check the specific characteristics of the data set, take a look on [kaggle](https://www.kaggle.com/datasets/enricogrimaldi/falls-vs-normal-activities)

We collected data using an Arduino model Nano 33 BLE Sense equipped with the necessary sensors for our experiment and engineered a small torso to be strapped to the back during data collection.

<img width="725" alt="IDE_collection" src="https://github.com/Engrima18/ToFall_orNot_toFall/assets/93355495/c90febb5-42a4-44f2-8643-1b38f62aada1">

The dataset is built in this way:

>- we have two different signals (accelerometer and gyroscope);
>- 3 axis for each signal (x, y, z);
>- the sampling rate of both sensors is fixed since, in order to simplify the data set and make it easier to manipulate, we collect data every 20 seconds, recording 100 values per second (for a total of 2000 timestamps). We then decide to only store and record 400 timestamps (values of the 2 signals in the 3 different axis) for each observation (experiment) with an effective sampling rate of 20 values per second ($\sim$ 20% of the sensors default collection rate);
>- in other words we end-up with a timeseries of length 400  for each axis of a patricular signal, i.e. a total of $400 \times 3 \times 2$ features;
>- each row will represent a specific observation (fall or not fall).

The class labels are:

>- fall: a severe fall with little movement after impact with the ground;
>- lfall: a lateral severe fall;
>- rfall: a reverse (back) severe fall;
>- light: a light fall, with movement afterthe impact or with a foiled full impact (landing on knees)
>- sit: "sitting" normal activity;
>- step: "going up and down the stairs" normal activity;
>- walk: "walking" normal activity.