# NumericCompFinalProj

## Event Class
0.25 deg granularity, 1 measurement per day. Should contain data from 1998-01-01 to 2022-12-31

Stores a 3D array of tuples (cmorph, humidity, surface_temp):
`data[date][lat][lon]`

Example Event:
```python
my_event = Event(    
    lat_min=37.0,
    lat_max=41.0,
    lon_min=-109.05,
    lon_max=-102.05,
    start_date="1998-01-01",
    end_date="1998-12-31",
)
# Creates an event with data ranging from 1998-01-01 to 1998-12-31 (year of 1998)
# Bounded to Colorado (lon and lat bound coords)
```

Accesors:
```python
# Basic event info
print(my_event)
# "Event(start_date=1998-01-01, end_date=1998-12-31, lat_min=37.0, lat_max=41.0, lon_min=250.95, lon_max=257.95)"

# Full 3d array of data. 
# data[date][lat][lon] = (cmorph, humidity, temperature)
data = my_event.get_data()

# 1d array of dates (for indexing data[date][...][...])
dates = my_event.get_dates()

# 1d array of latitudes (for indexing data[...][lat][...])
lats = my_event.get_lats()

# 1d array of longitutdes (for indexing data[...][...][lon])
lons = my_event.get_lons()
```